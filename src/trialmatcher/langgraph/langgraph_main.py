"""
Set up main multi-agent workflow for experiment

Use langraph:
    - https://langchain-ai.github.io/langgraph
    - https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/#graph
"""

import logging
import os
from datetime import datetime
from typing import Literal, Optional

from trialmatcher.trials import all_trial_criteria
from trialmatcher.utils import AzureClient, prep_vector_store
from trialmatcher.utils.schemas import TrialMatcherConfig, TrialMatcherState

from .construct_graph import build_graph

logger = logging.getLogger("trialmatcher")


def run_langgraph_trial_matcher(
    mrn: str,
    trial_id: str,
    run_config: TrialMatcherConfig,
    azure_client: AzureClient,
    cutoff_date: Optional[str] = None,
    current_date: Optional[str] = None,
    eligibility_ground_truth: Optional[Literal["eligible", "ineligible"]] = None,
):
    """main function to run the langgraph trial matcher workflow

    Args:
        mrn (str): mrn of patient
        trial_id (str): trial (protocol) of interest (e.g. '21-283')
        run_config (RAGConfig): Configuration for the RAG workflow
        azure_client (AzureClient): Azure client for managing Azure services
        cutoff_date (str, optional): date to filter notes. Notes created after this date will not be included. Must be in format 'mm-dd-yyyy', 'yyyy-mm-dd', 'mm/dd/yyyy', or 'yyyy/mm/dd'. Defaults to None.
        current_date (str, optional): current date for the trial. Passed to LLM in prompt so can be in any reasonable format. Defaults to None.
        eligibility_ground_truth (str, optional): ground truth for the trial. Used to evaluate performance. If provided, must be one of "eligible" or "ineligible". Defaults to None.
    """
    logger.info(f"Running trial matcher for MRN {mrn} and trial {trial_id}")

    # skip full run if already completed (avoid duplicating work)
    out_path = run_config.output_dir + f"/{mrn}_{trial_id}_output.json"
    if os.path.isfile(out_path):
        logger.info(f"Skipping run. Output file already exists: {out_path}")
        return

    assert (
        trial_id in all_trial_criteria
    ), f"Trial ID {trial_id} not found. Currently supported trials: {all_trial_criteria.keys()}"

    # prepare vectorstore
    vectorstore, vectorstore_tokens = prep_vector_store(
        mrn=mrn,
        embedding_model=azure_client.langchain_azure_openai_embeddings,
        cutoff_date=cutoff_date,
        run_config=run_config,
        disable_tqdm=not run_config.debug,
    )

    # build the graph
    graph = build_graph(run_config, vectorstore)

    # run the graph
    trial_criteria = all_trial_criteria[trial_id].copy()

    if run_config.debug_first_n:
        logger.warning(
            f"Running in debug mode: using only the first {run_config.debug_first_n} criteria"
        )
        trial_criteria = trial_criteria[: run_config.debug_first_n]

    initial_uncomplete = trial_criteria
    initial_complete = []

    n_total_criteria = len(initial_uncomplete)

    # handle vacuous criteria and criteria that require human review
    # iterate over a shallow copy so that we can safely remove elements without messing up indexing
    for c in initial_uncomplete[:]:
        if c.vacuous:
            logger.info(f"Handling vacuous criterion: {c.id}")
            c.determination = "met"
            c.explanation = {"rule": "Vacuous criterion"}
            initial_uncomplete.remove(c)
            initial_complete.append(c)
        elif c.requires_human_review:
            logger.info(f"Handling criterion requiring human review: {c.id}")
            c.determination = "unable to determine"
            c.explanation = {"rule": "Requires human review"}
            initial_uncomplete.remove(c)
            initial_complete.append(c)

    assert (
        len(initial_uncomplete) + len(initial_complete) == n_total_criteria
    ), "ERROR: Some criteria were not handled properly during initialization"

    initial_active = initial_uncomplete.pop(0)

    initial_state = TrialMatcherState(
        trial_id=trial_id,
        mrn=mrn,
        uncompleted_criteria=initial_uncomplete,
        active_criterion=initial_active,
        completed_criteria=initial_complete,
        n_total_criteria=n_total_criteria,
        final_determination=None,
        run_config=run_config,
        embedding_tokens=vectorstore_tokens,
        current_date=current_date,
        eligibility_ground_truth=eligibility_ground_truth,
        timestamp_start=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    # set high recursion limit - we expect lots of steps with complex workflow and lots of criteria
    final_state = graph.invoke(initial_state, {"recursion_limit": 500})

    return final_state
