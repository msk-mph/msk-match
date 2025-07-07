import logging

from trialmatcher.utils.schemas import TrialMatcherState


logger = logging.getLogger("trialmatcher")


def check_if_done(state: TrialMatcherState):
    logger.info("Checking if done")
    # if all criteria are completed, delegate to PI for final determination
    if not state.active_criterion:
        return "make_final_determination"
    # if there are uncompleted criteria, send back to trial coordinator to continue
    else:
        return "trial_coordinator"
