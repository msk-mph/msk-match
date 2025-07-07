import logging
import operator
from datetime import datetime
from typing import Annotated, Dict, List, Literal, Optional
from zoneinfo import ZoneInfo

from langchain_core.documents import Document
from pydantic import BaseModel

logger = logging.getLogger("trialmatcher")


class TrialMatcherConfig(BaseModel):
    description: Optional[str] = None  # description of the run
    # experiment name groups the runs together in the database
    # If experiment_name is None, assumed to be production
    experiment_name: Optional[str] = None
    # vectorstore config parameters
    embedding_model: Optional[str] = "text-embedding-3-large"
    k: Optional[int] = 6
    chunk_size: Optional[int] = 500
    chunk_overlap: Optional[int] = 50
    split_vectorstore_by_agent: Optional[Dict[str, List[str]]] = (
        None  # whether to split the vectorstore by agent. If so, provide a dict of {agent_name: [agent_keywords]}
    )
    exclude_note_keywords: Optional[List[str]] = (
        None  # keywords to exclude notes from vectorstore
    )

    # whether to use human feedback. Can disable for experiments etc.
    use_expert_feedback: Optional[bool] = True

    # LLM config parameters
    llm_model: Optional[str] = "gpt-4o-latest"
    openai_api_version: Optional[str] = "2024-08-01-preview"

    # exponential backoff config
    max_retries: int = 5
    base_wait: int = 1

    # method to use for making final determination
    final_determination_method: Optional[
        Literal["rule_based", "single_prompt", "chain_of_thought"]
    ] = "rule_based"

    check_explanations: Optional[bool] = (
        True  # whether to add a node to check explanations, refine them if necessary
    )
    graphrag_dir: Optional[str] = None  # path to working directory for graphrag

    # debug config
    debug: Optional[bool] = False
    debug_first_n: Optional[int] = None  # use only the first n criteria for debugging

    # inputs and outputs config
    output_dir: str

    # redis config
    redis_host: Optional[str] = None  # Redis host
    redis_port: Optional[int] = None  # Redis port

    # Data directory: A subdirectory here named `patient_vectorstores` will be used for writing/reading vectorstores to/from disk. A subdirectory here named `patient_records` will be used for saving patient EHR data.
    data_dir: str

    # run config
    # set time limit (in seconds) for each run
    timeout: Optional[int] = 600

    git_commit: Optional[str] = None  # git commit hash for the run


class Criterion(BaseModel):
    id: str
    criterion_text: str  # text of criterion copied directly from trial document
    criterion_type: Literal["inclusion", "exclusion"]
    vacuous: Optional[bool] = False  # always true, no need for LLM or human review
    requires_human_review: Optional[bool] = False  # always requires human review
    determination: Optional[Literal["met", "not met", "unable to determine"]]
    # explanations are dict {agent: explanation}
    explanation: Annotated[Dict[str, str] | None, operator.add] = None
    # this used internally for routing
    answered_by: Optional[str | List] = None
    # store the RAG documents used to answer the criterion
    rag_docs: Annotated[List[Document] | None, operator.add] = None


def active_criterion_reducer(current: Criterion, update: Criterion) -> Criterion:
    """
    Custom reducer to specify how to combine two active criteria
    This is used to enable branching in the graph
    Criteria are sent to different agents, each of which updates the explanations and rag docs
    Then this tells us how to combine the explanations and rag docs
    """
    # combine explanations
    logger.debug(
        f"Combining criteria-- Current: {current.id if current else 'None'} -- Update: {update.id if update else 'None'}"
    )
    logger.debug(
        f"Current exp: {current.explanation if current and current.explanation else 'None'} -- Update: {update.explanation if update and update.explanation else 'None'}"
    )
    # if the two criteria are the same, we can combine the explanations
    if update and current.id == update.id:
        if current.explanation == update.explanation:
            logger.debug("explanations are the same, no need to combine")
            return current
        if current.explanation and update.explanation is None:
            logger.debug("update explanation is None, using current")
            return current
        if update.explanation and current.explanation is None:
            logger.debug("current explanation is None, using update")
            return update
        if current.explanation and update.explanation:
            current.explanation = current.explanation | update.explanation

            # combine rag_docs but remove duplicates
            current_rag_ids = [doc.id for doc in current.rag_docs]
            update_rag_docs = [
                doc for doc in update.rag_docs if doc.id not in current_rag_ids
            ]
            current.rag_docs = current.rag_docs + update_rag_docs

            logger.debug("Reducer Combined explanations and rag docs")
            return current
        raise ValueError("Shouldn't get here")

    # if the two criteria are different, we can't combine them
    # in this case, we'll update to the new criterion
    # this allows principal_investigator to update the current criterion
    else:
        return update


class TrialMatcherState(BaseModel):
    trial_id: str
    mrn: str
    final_determination: Optional[Literal["eligible", "ineligible"]] = None
    # ground truth
    eligibility_ground_truth: Optional[Literal["eligible", "ineligible"]] = None
    # track token usage and cost
    input_tokens: Annotated[int, operator.add] = 0
    output_tokens: Annotated[int, operator.add] = 0
    embedding_tokens: Optional[int] = 0
    cost: Optional[float] = 0
    current_date: Optional[str] = None
    # metadata for tracking
    timestamp_start: Optional[str] = None  # timestamp for the run
    timestamp_end: Optional[str] = None  # timestamp for the end of the run
    time_elapsed_seconds: Optional[float] = None  # time elapsed for the run

    # used for storing the progress of the run
    # To start, most criteria should be uncompleted.
    # Then we move them to active, and then completed, as they are answered
    uncompleted_criteria: list[Criterion]
    active_criterion: Annotated[Criterion | None, active_criterion_reducer] = None
    n_total_criteria: Optional[int] = 0
    next_expert: Optional[List[str]] = None  # use for routing to the correct expert

    # config
    run_config: TrialMatcherConfig

    # final output of the run
    completed_criteria: list[Criterion]


class HumanFeedbackSingle(BaseModel):
    criterion_id: str
    human_determination: Literal["met", "not met", "unable to determine"]
    human_explanation: Optional[str]


class HumanFeedback(BaseModel):
    """
    Track when the human user decides to override model predictions and provide their own determination.
    This acts as a diff which can be applied to model outputs to get model + human collaborative output.
    Also track some stats on annotation time, etc.
    """

    time_duration: float
    # timestamp for the feedback. Use ISO format string.
    timestamp: str = datetime.now(ZoneInfo("America/New_York")).isoformat()
    trial_id: str
    mrn: str
    human_feedback: Optional[List[HumanFeedbackSingle]] = []
