import logging
from typing import Literal, Optional

from pydantic import BaseModel

from trialmatcher.utils import AzureClient
from trialmatcher.utils.schemas import TrialMatcherState

logger = logging.getLogger("trialmatcher")


class CheckExplanation(BaseModel):
    clarifying_question: Optional[str]
    next_step: Literal["continue", "query_and_refine"]


def check_explanation(state: TrialMatcherState) -> TrialMatcherState:
    """
    Node to check the explanation of the active criterion, after it has been generated.
    """
    logger.info("checking explanation")

    current_criterion = state.active_criterion

    azure_client = AzureClient(state.run_config)
    response = azure_client.chat_completions_parse(
        model=state.run_config.llm_model,
        messages=[
            {
                "role": "system",
                "content": """You are a helpful assistant working to help patients find out what clinical trials they are eligible for. You will be given a description of a particular criterion, along with one or more expert-generated explanations of whether the patient meets the criterion.

                You must check these explanations for accuracy and completeness. In most cases, the explanation will likely clear and correct and you can continue to the next step. However, if the explanation is unclear or contains inaccuracies, or is missing information, or makes errors in reasoning or logic, you should ask a clarifying question to help refine the explanation. Clarifying questions should be specific and related to funcamental concepts in medicine. For example, you might ask for more information about a specific test or procedure, or ask for clarification on a specific term or concept.
                
                ### Examples:
                
                # correct, logical explanation: continue
                Input:
                    Trial: 13-579
                    Criterion: Patients must have ER+ breast cancer
                    Criterion type: Inclusion
                    Explanation: The pathology report states that the tumor is "strongly ER+"
                Output:
                    clarifying_question: None
                    next_step: "continue"

                # explanation missing some information to make the correct conclusion: query_and_refine
                Input:
                    Trial: 12-345
                    Criterion: Patients are ineligible if they have metastatic breast cancer
                    Criterion type: Exclusion
                    Explanation: The radiology report notes that the cancer has spread to the axillary nodes. This is consistent with metastatic breast cancer.
                Output:
                    clarifying_question: does spread to the axillary nodes count as metastatic disease for breast cancer?
                    next_step: "query_and_refine"
                """,
            },
            {
                "role": "user",
                "content": f"Trial: {state.trial_id}\nCriterion: {current_criterion.criterion_text}\nCriterion type: {current_criterion.criterion_type}\nExplanation: {current_criterion.explanation}",
            },
        ],
        response_format=CheckExplanation,
        temperature=0.01,
    )
    response_data = response.choices[0].message.parsed
    logger.info(
        f"check_explanation token use-- input: {response.usage.prompt_tokens}, output: {response.usage.completion_tokens}"
    )
    next_step = response_data.next_step
    if next_step == "continue":
        logger.info("Explanation is clear and correct. Continuing to next step.")
        # delete model object
        # try to avoid problems with too many open files
        del azure_client
        logger.info("deleted azure_client object")
        return {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }

    # if not continue, then query and refine explanation
    clarifying_question = response_data.clarifying_question
    logger.info(
        f"Explanation is unclear or incorrect. Asking clarifying question: {clarifying_question}"
    )

    # query openai for information
    logger.info("Querying openai")
    response2 = azure_client.chat_completions_parse(
        model=state.run_config.llm_model,
        messages=[
            {
                "role": "system",
                "content": "You are an expert in cancer biology and all forms of oncology and management of cancer. You will be given a question, and you must provide an accurate, detailed, and comprehensive answer based on your expertise. The answer should be in the form of a paragraph or two, and should be as complete, information-dense, and informative as possible.",
            },
            {
                "role": "user",
                "content": f"Question: {clarifying_question}",
            },
        ],
        temperature=0.01,
    )
    query_answer = response2.choices[0].message.content

    # update the explanation with the new information
    logger.info("Refining explanation based on new information")
    response3 = azure_client.chat_completions_parse(
        model=state.run_config.llm_model,
        messages=[
            {
                "role": "system",
                "content": "You are an expert in cancer biology and all forms of oncology and management of cancer. You are workign in a clinical trial group to help determine if a patient is eligible for a clinical trial. You will be given a description of a particular criterion, along with one or more expert-generated explanations of whether the patient meets the criterion. You will also be given an additional query and answer, providing more information to help refine the explanation. Your task is to review the provided explanation, revise it as needed to address any logic errors or subject matter knowledge errors, and produce a final edited explanation that is clear, accurate, and complete.",
            },
            {
                "role": "user",
                "content": f"Criterion: {current_criterion.criterion_text}\Previous explanation: {current_criterion.explanation}\nQuery: {clarifying_question}\nAnswer: {query_answer}",
            },
        ],
        temperature=0.01,
    )
    revised_explanation = response3.choices[0].message.content

    logger.info(current_criterion.explanation)
    logger.info(type(current_criterion.explanation))

    current_criterion.explanation["revised"] = revised_explanation

    # keeping track of token usage
    input_tokens_total = response.usage.prompt_tokens + response3.usage.prompt_tokens
    if response2:
        input_tokens_total += response2.usage.prompt_tokens
    output_tokens_total = (
        response.usage.completion_tokens + response3.usage.completion_tokens
    )
    if response2:
        output_tokens_total += response2.usage.completion_tokens

    # delete model object
    # try to avoid problems with too many open files
    del azure_client
    logger.info("deleted azure_client object")

    return {
        "input_tokens": input_tokens_total,
        "output_tokens": output_tokens_total,
        "current_criterion": current_criterion,
    }
