from typing import Literal
from pydantic import BaseModel
import logging

from trialmatcher.utils.schemas import TrialMatcherState
from trialmatcher.utils import retry_with_exponential_backoff, AzureClient

logger = logging.getLogger("trialmatcher")


class FinalDetermination(BaseModel):
    determination: Literal["eligible", "ineligible"]


def get_final_determination_node(method: str):
    if method == "rule_based":
        return final_determination_rule_based
    if method == "single_prompt":
        return final_determination_single_prompt
    if method == "chain_of_thought":
        return final_determination_COT
    raise ValueError(
        f"Invalid method: {method}. Should be one of 'rule_based', 'single_prompt', 'chain_of_thought'"
    )


def final_determination_rule_based(state: TrialMatcherState):
    logger.info("Making final determination: rules-based")
    n = len(state.completed_criteria) + len(state.uncompleted_criteria)
    if state.active_criterion:
        n += 1
    assert (
        n == state.n_total_criteria
    ), f"ERROR: some criteria have been lost! Expected {state.n_total_criteria}, found {n}"
    logger.debug(f"num criteria entering final determination: {n}")
    # use rule-based logic to make final determination
    # check that no eligibility criteria are unmet and no exclusion criteria are met
    for crit in state.completed_criteria:
        if crit.determination == "unable to determine":
            continue
        if crit.criterion_type == "inclusion" and crit.determination == "not met":
            state.final_determination = "ineligible"
            return state
        if crit.criterion_type == "exclusion" and crit.determination == "met":
            state.final_determination = "ineligible"
            return state
    state.final_determination = "eligible"
    return state


@retry_with_exponential_backoff(max_retries=5, base_wait=1)
def final_determination_single_prompt(state: TrialMatcherState) -> str:
    """
    Make final determination by putting all criteria and their explanations into a single prompt
    """
    logger.info("Making final determination: single prompt")
    # prepare prompt
    prompt = ""
    for criterion in state.completed_criteria:
        prompt += (
            f"{criterion.id}:{criterion.criterion_text}\n{criterion.explanation}\n\n"
        )

    class FinalDeterminationExplanation(FinalDetermination):
        explanation: str

    azure_client = AzureClient(state.run_config)
    response = azure_client.chat_completions_parse(
        model=state.run_config.llm_model,
        messages=[
            {
                "role": "system",
                "content": "You are the principal investigator for a clinical trial. Your team is evaluating a patient to determine whether or not they are eligible for the trial. They have already looked at each of the inclusion and exclusion criteria, extracting relevant information from the patient's medical records and making some explanations for each criterion. Now, they need your final determination on whether the patient is eligible or not. Please review the information provided, identify and correct any mistakes made, and make a correct final determination. A patient is ineligible if there is at least one inclusion criteria which they do not meet, or if they meet at least one of the exclusion criteria. Criteria which are 'unable to determine', for example they need more information or require human review, should not be considered when making this determination. Also give a brief (1 sentence max) explanation for your determination.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format=FinalDeterminationExplanation,
        temperature=0.4,
    )
    final_determination = response.choices[0].message.parsed.determination
    state.final_determination = final_determination

    # delete model object
    # try to avoid problems with too many open files
    del azure_client
    logger.info("deleted azure_client object")

    return state


@retry_with_exponential_backoff(max_retries=5, base_wait=1)
def final_determination_COT(state: TrialMatcherState) -> str:
    """
    Make final determination by putting all criteria and their explanations into a single prompt, then using chain of thought reasoning.
    Based on prompts from OncoLLM: https://arxiv.org/pdf/2404.15549v1
    note the prompt details were posted in preprint version 1 on arxiv and then removed in version 2
    slightly modified to fit our workflow, which is slightly different than in their paper
    """
    logger.info("Making final determination: chain of thought")
    onco_llm_prompt = """
    You are an Experienced Clinical Trial Matching Assistant, your task is to accurately determine clinical trial eligibility for a cancer patient based on their Electronic Health Record (EHR) data. The final determination should be made with an 'Eligible' or 'Not Eligible' response. 

    To represent the degree of certainty in your answer, provide a numerical confidence score ranging from 1 (not confidence) to 5 (highly confident). 

    Your team has already looked at each of the inclusion and exclusion criteria, extracting relevant information from the patient's medical records and making some explanations for each criterion. You will be provided with this information to reason your final answer for the eligibility. Ensure that you are examining all the aspects of the patient documents and using medical reasoning to arrive at the final answer. Also you must ensure to consider the temporal aspects of the criteria as well, using the provided CURRENT DATE as the current date for all your analysis. The success of your job depends on it, therefore take the necessary time to reason thoroughly. 

    In some cases, the explanations can have indirect information. For example, if there's some information about the patient's TNM staging according to AJCC (e.g., cT3N0M0 or T3, N0, M0), where T describes the size of the tumor and any spread of cancer into nearby tissue; N describes the spread of cancer to nearby lymph nodes; and M describes metastasis (spread of cancer to other parts of the body), then make sure to use these indirect information as well to form your answers. In some instances, if a patient's condition is queried and the lab/test reports used to assess that condition are provided without indicating the presence of the disease in the patient, it is acceptable to presume that the patient does not have that condition. In your response, please cite the source of your information. Also, provide a detailed reasoning/explanation of your answer. Ensure your reasoning is logical, clear, succinct and medically correct.

    A patient is ineligible if there is at least one inclusion criteria which they do not meet, or if they meet at least one of the exclusion criteria. Criteria which are 'unable to determine', for example they need more information or require human review, should not be considered when making this determination. 

    ## Go through each of the criteria and explanations provided by the team, one by one. To ensure you answer the criteria very very accurately, you can follow the following step-by-step method to do the reasoning:
    1. Understand the provided criteria properly, work out a strategy to answer a criteria like this. For some questions, temporal aspects is also important, take your time and make an informed decision whether to consider temporal aspects or not.
    2. Understand the meaning and relationship between all the specified medical/clinical terms in the criteria and explanations.
    3. Now work out a step-by-step logical deduction to answer the criterion verbally. For example:
        - For the question ”Is the tumor size <10 cm?”, then work out like this: ”The patient has tumor size of 5 cm (CHUNK ID) and 5 cm is less than 10 cm, which means tumor size <10 cm.
        - For the question ”Does the patient have breast cancer?”, then work out like this: ”The patient's primary site of cancer is Nipple (document citation). Since Nipple is the primary site of breast cancer, which mean the patient has breast cancer.
    4. Now double check the explanation provided by the team, and if there are any discrepancies or errors, correct them.
    5. After checking all of the criteria, make a final decision on the patient's eligibility for the trial: 'Eligible' or 'Not Eligible'.
    6. Give a short (1 sentence max) explanation for your determination.
    7. Finally, provide a confidence score between 1-5, based on how confident you are in your answer.
    """

    # https://platform.openai.com/docs/guides/structured-outputs#chain-of-thought
    class Step(BaseModel):
        explanation: str
        output: str

    class FinalDeterminationReasoning(FinalDetermination):
        steps: list[Step]
        confidence: int
        explanation: str

    # prepare prompt
    prompt = f"""< CURRENT_DATE >
    {state.current_date}
    </ CURRENT_DATE >
    """
    for criterion in state.completed_criteria:
        prompt += f"""\n< CRITERION >
        ID: {criterion.id}
        TEXT: {criterion.criterion_text}
        EXPLANATION: {criterion.explanation}
        </ CRITERION >
        """

    azure_client = AzureClient(state.run_config)
    response = azure_client.chat_completions_parse(
        model=state.run_config.llm_model,
        messages=[
            {
                "role": "system",
                "content": onco_llm_prompt,
            },
            {"role": "user", "content": prompt},
        ],
        response_format=FinalDeterminationReasoning,
        temperature=0.4,
    )
    final_determination = response.choices[0].message.parsed.determination
    state.final_determination = final_determination

    # delete model object
    # try to avoid problems with too many open files
    del azure_client
    logger.info("deleted azure_client object")

    return state
