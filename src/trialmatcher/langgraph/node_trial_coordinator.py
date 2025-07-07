import logging

from trialmatcher.utils.schemas import TrialMatcherState

logger = logging.getLogger("trialmatcher")


def trial_coordinator(state: TrialMatcherState):
    logger.info("Running trial coordinator")
    logger.info(f"Active criterion: {state.active_criterion.id}")
    logger.info(
        f"token usage stats from state: input: {state.input_tokens}, output: {state.output_tokens}"
    )
    return
