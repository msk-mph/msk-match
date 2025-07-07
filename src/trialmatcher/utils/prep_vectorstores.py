from datetime import datetime
import os
from typing import Tuple
from tqdm import tqdm
import time
import tiktoken
import logging
import re
import math
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings

import trialmatcher.utils
from trialmatcher.utils.schemas import TrialMatcherConfig


logger = logging.getLogger("trialmatcher")


def prep_vector_store(
    *,
    mrn: str,
    embedding_model: Embeddings,
    cutoff_date: str = None,
    run_config: TrialMatcherConfig,
    disable_tqdm: bool = False,
) -> Tuple[InMemoryVectorStore, int]:
    """Prepare a vector store for a given patient's records. Load from disk if it exists, otherwise create it.

    Args:
        mrn (str): Patient's MRN.
        embedding_model (langchain_core.embeddings.Embeddings): Embedding model to use for creating the vectorstore. E.g. AzureOpenAIEmbeddings.
        cutoff_date (str, optional): Cutoff date for which records to include. Records after this date will be dropped. Defaults to None.
        run_config (TrialMatcherConfig): Configuration for the run.
        disable_tqdm (bool, optional): Whether to display a progress bar for tracking vectorstore creation. Defaults to False.

    Returns:
        InMemoryVectorStore: vectorstore of patient records
        int: number of tokens used in creating the vectorstore
    """

    parameterized_file_name = f"vectorstore_{mrn}_{embedding_model.deployment}_chunk-size-{run_config.chunk_size}_chunk-overlap-{run_config.chunk_overlap}.pkl"

    if run_config.data_dir:
        parameterized_file_path = (
            f"{run_config.data_dir}/patient_vectorstores/{parameterized_file_name}"
        )

    # check if vector store already exists, load if it does
    if run_config.data_dir:
        if os.path.isfile(parameterized_file_path):
            try:
                vectorstore = InMemoryVectorStore.load(
                    path=parameterized_file_path, embedding=embedding_model
                )
                logger.info(f"loaded vectorstore from file: {parameterized_file_path}")
                return vectorstore, 0  # no tokens used in loading from disk
            except Exception:
                logger.error(
                    f"Error loading vectorstore from file: {parameterized_file_path}"
                )
                logger.error("falling back to creating vectorstore from scratch")

    data_structured, data_unstructured = trialmatcher.utils.process_dumped_ehr_data(
        mrn, data_dir=f"{run_config.data_dir}/patient_records"
    )

    # filter out documents that we don't want to include
    data_unstructured = data_unstructured[
        ~data_unstructured["content"].str.startswith(
            "This document is intentionally left blank."
        )
    ]
    if run_config.exclude_note_keywords:
        data_unstructured = data_unstructured.loc[
            ~data_unstructured.type.str.lower().str.contains(
                "|".join(run_config.exclude_note_keywords)
            )
        ].reset_index(drop=True)

    # filter documents with date after cutoff
    if cutoff_date:
        logger.info(f"Filtering out documents after cutoff date: {cutoff_date}")
        logger.info(f"Original number of documents: {len(data_unstructured)}")
        date_filter = try_parsing_date(cutoff_date)
        data_unstructured = data_unstructured[
            data_unstructured["procedure_date"] <= date_filter
        ].reset_index(drop=True)
        logger.info(f"Number of documents after filtering: {len(data_unstructured)}")

    # data column into string if needed, so that it can be serialized
    try:
        data_unstructured["procedure_date"] = data_unstructured[
            "procedure_date"
        ].dt.strftime("%Y-%m-%d")
    except AttributeError:
        pass

    # remove newlines and extra whitespace before embedding
    data_unstructured["content"] = data_unstructured["content"].apply(
        lambda x: re.sub(r"\s+", " ", x.strip())
    )

    loader = DataFrameLoader(data_unstructured, page_content_column="content")
    ehr = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=run_config.chunk_size, chunk_overlap=run_config.chunk_overlap
    )
    splits = text_splitter.split_documents(ehr)

    vectorstore = InMemoryVectorStore(embedding=embedding_model)

    # get tokenizer for gpt-4o
    tokenizer = tiktoken.encoding_for_model("gpt-4o")
    num_tokens = 0

    # log progress of vectorstore creation
    total = len(splits)
    logger.info(f"Beginning vectorstore creation with {total} splits")
    # Compute milestone indices for 25%, 50%, 75%, and 100%
    milestones = {math.ceil(total * pct) for pct in [0.25, 0.5, 0.75, 1.0]}

    for i, split in enumerate(tqdm(splits, disable=disable_tqdm), start=1):
        # track token use: https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken
        num_tokens += len(tokenizer.encode(split.page_content))
        vectorstore.add_documents(documents=[split], embedding=embedding_model)
        time.sleep(0.001)

        if i in milestones:
            progress = (i / total) * 100
            logger.info(f"Progress: {progress:.0f}% complete")

    # save for later
    if run_config.data_dir:
        # create dir if it doesn't exist
        os.makedirs(run_config.data_dir, exist_ok=True)
        vectorstore.dump(path=parameterized_file_path)
        logger.info(f"Saved vectorstore to file: {parameterized_file_path}")
    return vectorstore, num_tokens


# From https://stackoverflow.com/a/23581184
def try_parsing_date(text: str) -> datetime:
    """parse a date from a string with multiple possible formats"""
    supported_formats = ["%m-%d-%Y", "%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d"]
    for fmt in supported_formats:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    raise ValueError(
        f"no valid date format found for input: {text}\nSupported formats: {supported_formats}"
    )


def split_vectorstore_by_agent(
    vectorstore: InMemoryVectorStore,
    agent_names_keywords: dict[str, list[str]],
) -> dict[str, InMemoryVectorStore]:
    """
    Takes a vectorstore and splits it into multiple vectorstores, one for each agent
    based on the agent_names_keywords dictionary

    Args:
        vectorstore (InMemoryVectorStore): The vectorstore to split
        agent_names_keywords (dict[str, list[str]]): A dictionary of {agent_name: [keywords]} mapping agent names to a list of keywords.

    Returns:
        dict[str, InMemoryVectorStore]: A dictionary mapping agent names to their corresponding vectorstores
    """

    agent_names_vectorstores = {}

    for agent_name, agent_kw in agent_names_keywords.items():
        # get only the notes that are relevant to the agent
        agent_kw_pattern = "|".join(agent_kw)

        # find all the documents that contain the agent keywords
        agent_docs = [
            d
            for d in vectorstore.store.values()
            if re.search(agent_kw_pattern, d["metadata"]["type"], re.IGNORECASE)
        ]

        if not agent_docs:
            logger.info(
                f"No data for {agent_name} with keywords '{agent_kw_pattern}' ({agent_kw})"
            )
            continue

        # create a new vectorstore for the agent
        agent_vectorstore = InMemoryVectorStore(embedding=vectorstore.embedding)

        # add docs directly to the store, so that they aren't re-embedded
        for doc in agent_docs:
            agent_vectorstore.store[doc["id"]] = doc

        agent_names_vectorstores[agent_name] = agent_vectorstore

    # add generalist agent for notes that don't fit into any of the specialist categories
    all_agent_kw_pattern = "|".join(
        [kw for kw_list in agent_names_keywords.values() for kw in kw_list]
    )
    generalist_docs = [
        d
        for d in vectorstore.store.values()
        if not re.search(all_agent_kw_pattern, d["metadata"]["type"], re.IGNORECASE)
    ]
    if generalist_docs:
        generalist_vectorstore = InMemoryVectorStore(embedding=vectorstore.embedding)
        for doc in generalist_docs:
            generalist_vectorstore.store[doc["id"]] = doc
        agent_names_vectorstores["generalist"] = generalist_vectorstore
    else:
        logger.info("No data for generalist")

    return agent_names_vectorstores
