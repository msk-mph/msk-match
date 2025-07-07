import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import AzureChatOpenAI

from trialmatcher import config
from trialmatcher.utils import RedisManager, retry_with_exponential_backoff
from trialmatcher.utils.schemas import TrialMatcherState

logger = logging.getLogger("trialmatcher")


def consult_agent(
    agent_name: str, vectorstore: InMemoryVectorStore, state: TrialMatcherState
):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.

    Args:
        agent_name (str): The name of the agent
        vectorstore (InMemoryVectorStore): The vectorstore for the agent's notes
        state (TrialMatcherState): The current state

    Returns:
        dict: The updated state with the agent response updated in active_criterion explanation
    """
    logger.info(f"Consulting agent: {agent_name}")

    # need to make a copy because otherwise the different agents will overwrite each other's explanations
    active_criterion = state.active_criterion.model_copy()

    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": state.run_config.k}
    )

    # get the human input data
    if state.run_config.use_expert_feedback:
        if state.run_config.redis_host and state.run_config.redis_port:
            # load feedback from redis
            redis_manager = RedisManager(
                host=state.run_config.redis_host, port=state.run_config.redis_port
            )
            logger.info("Loading kb from redis")
            human_input_data = redis_manager.get_human_feedback()
        else:
            # load feedback from disk
            kb_path = f"{state.run_config.data_dir}kb.txt"
            logger.info(f"Loading kb from disk: {kb_path}")
            with open(kb_path, "r") as f:
                lines = f.readlines()
            human_input_data = [line.strip() for line in lines]

        logger.info(f"retrieved human input data with {len(human_input_data)} entries")
        human_input_data = "\n".join(human_input_data)
    else:
        human_input_data = "No human input data available."
        logger.info("Skipping human use_expert_feedback data in prompt")

    prompt_template = f"""You are a {agent_name}. You have been asked to assess a patient for eligibility for a clinical trial. You have access to query the relevant reports from your specialty for information. You always cite which document you get your information from, in the format '([report type] - [date])'. If a question is not relevant to your expertise, you allow another specialist to answer. If you don't have enough information to answer the question, you write 'unable to determine'. You will be given a set of human input data provided by human experts to guide decision-making. When expert data is relied upon to make a determination, you should make it clear that you are incorporating the expert feedback in your decision-making process. You should always defer to the expert data when it is available. Think step-by-step before giving your final explanation for your answer, citing source documents. Keep your answer to one paragraph or less whenever possible.

    ### Expert data: 
    {human_input_data}
    """
    if state.current_date:
        prompt_template += f"\nToday's date: {state.current_date}"

    if state.run_config.debug:
        # print(f"prompt template: {prompt_template}")
        pass

    prompt_template += """\nCriterion: {criterion_text}\nContext: {context}"""
    # if active_criterion.explanation:
    #     prompt_template += f"\nBuild on the previous explanation, which was judged to be insufficient: {active_criterion.explanation}"
    prompt_template += "\nAnswer:"

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "criterion_text"],
    )
    # logger.debug(f"Prompt: {prompt_template}")
    model = AzureChatOpenAI(
        openai_api_version=state.run_config.openai_api_version,
        azure_deployment=state.run_config.llm_model,
        api_key=config.AZURE_OPENAI_API_KEY,
        azure_endpoint=config.AZURE_OPENAI_API_ENDPOINT,
        temperature=0,
    )

    def format_docs(docs, criterion=active_criterion):
        # format all the retrieved documents into a single string
        out = ""
        for doc in docs:
            out += f"type: {doc.metadata['type']}"
            if doc.metadata["sub_type"]:
                out += f", subtype: {doc.metadata['sub_type']}"
            out += f", date: {doc.metadata['procedure_date']}\n"
            out += doc.page_content + "\n\n"

        # save the RAG docs to the state so we can access them later
        criterion.rag_docs = docs
        logger.debug(f"number of RAG docs: {len(criterion.rag_docs)}")

        return out

    rag_chain = (
        {
            "context": retriever | format_docs,
            "criterion_text": RunnablePassthrough(),
        }
        | prompt
        | model
    )

    # delete model object
    # try to avoid problems with too many open files
    del model
    logger.info("deleted model object")

    # wrap the chain with retry decorator to handle OpenAI rate limiting
    invoke_wrapper = retry_with_exponential_backoff()(rag_chain.invoke)
    response = invoke_wrapper(active_criterion.criterion_text)

    logger.info(
        f"{agent_name} token use-- input: {response.usage_metadata['input_tokens']}, output: {response.usage_metadata['output_tokens']}"
    )

    # parse the response
    parsed_response = {agent_name: StrOutputParser().invoke(response)}
    logger.debug(f"Parsed response: {parsed_response}")

    # parse the response
    active_criterion.explanation = parsed_response
    assert active_criterion.explanation == parsed_response
    logger.debug(
        f"Active criterion explanation after updating with parsed response: {active_criterion.explanation}"
    )
    # update active criterion
    return {
        "active_criterion": active_criterion,
        "input_tokens": response.usage_metadata["input_tokens"],
        "output_tokens": response.usage_metadata["output_tokens"],
    }
