from typing import Literal
from pydantic import BaseModel
import logging

from trialmatcher.utils import AzureClient
from trialmatcher.utils.schemas import TrialMatcherState


logger = logging.getLogger("trialmatcher")


class PIDetermination(BaseModel):
    determination: Literal["met", "not met", "unable to determine"]


def principal_investigator(state: TrialMatcherState):
    # adjudicate the criterion-level response as "met", "not met", or "unable to determine"
    logger.info("Running principal investigator")
    logger.debug(
        f"entering PI agent: uncomp: {len(state.uncompleted_criteria)}. comp: {len(state.completed_criteria)}. active: {state.active_criterion.id if state.active_criterion else 'None'}"
    )
    current_criterion = state.active_criterion
    logger.debug(f"Current criterion explanation: {current_criterion.explanation}")
    azure_client = AzureClient(state.run_config)
    response = azure_client.chat_completions_parse(
        model=state.run_config.llm_model,
        messages=[
            {
                "role": "system",
                "content": """You are the principal investigator for a clinical trial. You will be given a description of a particular criterion, along with one or more expert-generated explanations of whether the patient meets the criterion. 
                
                You must synthesize these different explanations and make a final adjudication as 'met' or 'not met'. For consistency, focus your answer on the criteria, NOT the overall trial eligibility (that will be determined at a later step). For example, if an exclusion criterion is met, meaning that the patient is ineligible, you still write 'met'. Similarly, if an exclusion criterion is not met, meaning that the patient may still be eligible, you write 'not met'. Be careful and precise in your logic. If a criterion is not applicable to the patient, then you answer 'met' if it is an inclusion criterion, and 'not met' if it is an exclusion criterion (so that the criterion is not a barrier to eligibility). If you decide that there is truly not enough information to make a determination, you write 'unable to determine' to flag the criterion for manual review by a human. However, you try to avoid this as much as possible, and give helpful 'met' and 'not met' answers.
                """,
            },
            {
                "role": "user",
                "content": f"Trial: {state.trial_id}\nCriterion: {current_criterion.criterion_text}\nCriterion type: {current_criterion.criterion_type}\nExplanation: {current_criterion.explanation}",
            },
        ],
        response_format=PIDetermination,
        temperature=0.01,
    )
    logger.info(
        f"PI token use-- input: {response.usage.prompt_tokens}, output: {response.usage.completion_tokens}"
    )
    determination = response.choices[0].message.parsed.determination

    current_criterion.determination = determination

    # delete model object
    # try to avoid problems with too many open files
    del azure_client
    logger.info("deleted azure_client object")

    return {
        "active_criterion": current_criterion,
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
    }
