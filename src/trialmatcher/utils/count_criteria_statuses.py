from trialmatcher.utils.schemas import TrialMatcherState


def count_criteria_statuses(state: dict | TrialMatcherState) -> dict:
    """
    Compute top-level numbers for criteria statuses

    Qualifying: inclusion criteria met, exclusion criteria not met
    Disqualifying: inclusion criteria not met, exclusion criteria met
    Unable to determine: criteria that are unable to be determined

    Args:
        state (TrialMatcherState): The current state of the trial matcher

    Returns:
        dict: A dictionary with counts of each status
    """

    if isinstance(state, str):
        # If state is a dictionary, convert it to a TrialMatcherState object
        state = TrialMatcherState.model_validate_json(state)

    counts = {"qualifying": 0, "disqualifying": 0, "unable to determine": 0}
    for crit in state.completed_criteria:
        if crit.determination == "unable to determine":
            counts["unable to determine"] += 1
        if (crit.criterion_type == "inclusion" and crit.determination == "not met") or (
            crit.criterion_type == "exclusion" and crit.determination == "met"
        ):
            counts["disqualifying"] += 1
        if (crit.criterion_type == "inclusion" and crit.determination == "met") or (
            crit.criterion_type == "exclusion" and crit.determination == "not met"
        ):
            counts["qualifying"] += 1
    return counts
