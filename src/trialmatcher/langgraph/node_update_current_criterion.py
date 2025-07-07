from trialmatcher.utils.schemas import TrialMatcherState


def update_current_criterion(state: TrialMatcherState):
    # move the active criterion to the completed criteria
    # get the next criterion and set it as the active criterion
    # update the lists in the state
    # this node should precede checking if done each iteration
    current_criterion = state.active_criterion
    new_active = (
        state.uncompleted_criteria.pop(0) if state.uncompleted_criteria else None
    )
    new_completed = state.completed_criteria + [current_criterion]
    return {
        "uncompleted_criteria": state.uncompleted_criteria,
        "active_criterion": new_active,
        "completed_criteria": new_completed,
    }
