"""
Build graph structure, depending on the configuration
"""

import logging
from langgraph.graph import END, START, StateGraph
from langchain_core.vectorstores import InMemoryVectorStore
from pydantic import BaseModel, field_validator
from typing import List, Literal

from trialmatcher.utils.schemas import TrialMatcherConfig, TrialMatcherState
from trialmatcher.utils import split_vectorstore_by_agent, AzureClient
from .node_trial_coordinator import trial_coordinator
from .node_save_results import save_results
from .node_principal_investigator import principal_investigator
from .node_make_final_determination import get_final_determination_node
from .node_consult_agent import consult_agent
from .router_check_if_done import check_if_done
from .node_check_explanation import check_explanation
from .node_update_current_criterion import update_current_criterion


logger = logging.getLogger("trialmatcher")


def build_graph(
    run_config: TrialMatcherConfig, vectorstore: InMemoryVectorStore
) -> StateGraph:
    """Constructs the langgraph Graph, based on the configuration

    Args:
        run_config (TrialMatcherConfig): configuration for the run
        vectorstore (InMemoryVectorStore): Vectorstore of patient records

    Returns:
        StateGraph: compiled langgraph object
    """
    if run_config.split_vectorstore_by_agent:
        try:
            graph = _build_graph_multi_expert_branching(run_config, vectorstore)
        except Exception as e:
            logger.error(f"Error building graph with multiple experts: {e}")
            logger.error("Trying to fall back to single expert graph")
            graph = _build_graph_single_rag(run_config, vectorstore)

    else:
        graph = _build_graph_single_rag(run_config, vectorstore)

    logger.info("Graph compiled")
    logger.info("Logging graph structure diagram:")
    logger.info(graph.get_graph().draw_mermaid())

    return graph


def _build_graph_single_rag(
    run_config: TrialMatcherConfig, vectorstore: InMemoryVectorStore
) -> StateGraph:
    """Builds a computation graph with a single RAG node, with access to all the notes

    Args:
        run_config (TrialMatcherConfig): configuration for the run
        vectorstore (InMemoryVectorStore): Vectorstore of patient records

    Returns:
        StateGraph: compiled langgraph object
    """
    workflow = StateGraph(TrialMatcherState)
    # add nodes
    workflow.add_node("trial_coordinator", trial_coordinator)
    workflow.add_node("principal_investigator", principal_investigator)
    workflow.add_node(
        "make_final_determination",
        get_final_determination_node(run_config.final_determination_method),
    )
    workflow.add_node("save_results", save_results)

    def create_expert_node():
        return lambda state: consult_agent(
            agent_name="expert",
            vectorstore=vectorstore,
            state=state,
        )

    workflow.add_node("expert", create_expert_node())
    workflow.add_node("update_current_criterion", update_current_criterion)

    # Set the entrypoint ie which node is the first one called
    workflow.add_edge(START, "trial_coordinator")
    workflow.add_edge("trial_coordinator", "expert")
    workflow.add_edge("expert", "principal_investigator")

    if run_config.check_explanations:
        workflow.add_node("check_explanation", check_explanation)
        workflow.add_edge("principal_investigator", "check_explanation")
        workflow.add_edge("check_explanation", "update_current_criterion")
    else:
        workflow.add_edge("principal_investigator", "update_current_criterion")

    workflow.add_conditional_edges(
        "update_current_criterion",
        check_if_done,
        path_map={
            "make_final_determination": "make_final_determination",
            "trial_coordinator": "trial_coordinator",
        },
    )
    workflow.add_edge("make_final_determination", "save_results")
    workflow.add_edge("save_results", END)

    # now compile the graph
    graph = workflow.compile()
    return graph


def _build_graph_multi_expert_branching(
    run_config: TrialMatcherConfig, vectorstore: InMemoryVectorStore
) -> StateGraph:
    """
    Build a graph with multiple experts, each with their own vectorstore
    Routes queries to the correct expert(s)

    Args:
        run_config (TrialMatcherConfig): configuration for the run
        vectorstore (InMemoryVectorStore): Vectorstore of patient records

    Returns:
        StateGraph: compiled langgraph object
    """
    # split the vectorstore by agent
    agent_name_to_vectorstores = split_vectorstore_by_agent(
        vectorstore=vectorstore,
        agent_names_keywords=run_config.split_vectorstore_by_agent,
    )

    if not agent_name_to_vectorstores:
        logger.error("agent_name_to_vectorstores is empty. Cannot create graph.")
        logger.error(f"vectorstore size: {len(vectorstore.store)}")

        raise ValueError(
            "agent_name_to_vectorstores is empty. Cannot create Literal with no valid choices."
        )

    ## conditional router
    # Define the function that determines which expert to route to
    # need to define this inside the main function because the options for experts might change betwwen patients
    # Defining the allowed choices at runtime doesn't work for type checking, but works for structured outputs
    expert_choices = tuple(agent_name_to_vectorstores.keys())
    logger.info(f"Expert choices: {expert_choices}")

    class ExpertChoice(BaseModel):
        expert: List[Literal[expert_choices]]  # type: ignore

        # make sure the expert list is unique
        # https://docs.pydantic.dev/latest/concepts/validators/#validation-of-default-values
        @field_validator("expert")
        @classmethod
        def make_unique(cls, val):
            return list(set(val))

    # node to consult an expert
    def consult_expert(
        state: TrialMatcherState,
    ) -> ExpertChoice:
        azure_client = AzureClient(state.run_config)
        response = azure_client.chat_completions_parse(
            model=state.run_config.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are the coordinator for a clinical trial. You will be given a criterion and you must determine which expert to delegate the task to. You can send to a specialist, or to a generalist if the question is not relevant to any specialist. You can also send to multiple experts if the question is relevant to multiple specialties.",
                },
                {"role": "user", "content": state.active_criterion.criterion_text},
            ],
            response_format=ExpertChoice,
            temperature=0.01,
        )
        next_expert = response.choices[0].message.parsed.expert
        logger.info(f"Choosing which expert(s) to consult: {next_expert}")

        if not next_expert:
            if "generalist" in expert_choices:
                next_expert = ["generalist"]
                logger.info("No specific expert chosen, defaulting to generalist")
            else:
                next_expert = expert_choices
                logger.info(
                    f"No specific expert chosen, generalist not available, defaulting to all available expert: {expert_choices}"
                )

        logger.info(
            f"consult router token use-- input: {response.usage.prompt_tokens}, output: {response.usage.completion_tokens}"
        )
        state.active_criterion.answered_by = next_expert
        # delete model object
        # try to avoid problems with too many open files
        del azure_client
        logger.info("deleted azure_client object")
        return {
            "active_criterion": state.active_criterion,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }

    # conditional edge
    def route_consult_expert(state: TrialMatcherState):
        return state.active_criterion.answered_by

    workflow = StateGraph(TrialMatcherState)
    # add nodes
    workflow.add_node("trial_coordinator", trial_coordinator)
    workflow.add_node("consult_expert", consult_expert)
    workflow.add_node("principal_investigator", principal_investigator)
    workflow.add_node("update_current_criterion", update_current_criterion)
    workflow.add_node(
        "make_final_determination",
        get_final_determination_node(run_config.final_determination_method),
    )
    workflow.add_node("save_results", save_results)

    if run_config.check_explanations:
        workflow.add_node("check_explanation", check_explanation)
        workflow.add_edge("principal_investigator", "check_explanation")
        workflow.add_edge("check_explanation", "update_current_criterion")
    else:
        workflow.add_edge("principal_investigator", "update_current_criterion")

    workflow.add_conditional_edges(
        "update_current_criterion",
        check_if_done,
        path_map={
            "make_final_determination": "make_final_determination",
            "trial_coordinator": "trial_coordinator",
        },
    )

    # Use a helper function to create node for each agent.
    # Can't use lambda directly in add_node() because of closure issues.
    # Without this, all lambdas would capture the same `agent_name` (the last value in the loop),
    # because Python evaluates loop variables lazily in closures.
    def create_agent_node(agent_name):
        return lambda state: consult_agent(
            agent_name=agent_name,
            vectorstore=agent_name_to_vectorstores[agent_name],
            state=state,
        )

    for agent_name in agent_name_to_vectorstores.keys():
        workflow.add_node(agent_name, create_agent_node(agent_name))
        # each expert sends its output to the PI
        workflow.add_edge(agent_name, "principal_investigator")

    workflow.add_conditional_edges(
        "consult_expert",
        route_consult_expert,
        path_map={exp: exp for exp in expert_choices},
    )

    # Set the entrypoint ie which node is the first one called
    workflow.add_edge(START, "trial_coordinator")
    workflow.add_edge("trial_coordinator", "consult_expert")
    workflow.add_edge("make_final_determination", "save_results")
    workflow.add_edge("save_results", END)

    # now compile the graph
    graph = workflow.compile()

    return graph
