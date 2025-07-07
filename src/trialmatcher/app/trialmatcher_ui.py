import redis.connection
import streamlit as st
import time
import re
import redis
import sys

import trialmatcher
from trialmatcher.trials import all_trial_criteria
from trialmatcher.utils.schemas import (
    TrialMatcherState,
    Criterion,
    HumanFeedbackSingle,
    HumanFeedback,
)
from trialmatcher.langgraph.node_make_final_determination import (
    final_determination_rule_based,
)
from trialmatcher.utils import RedisManager, count_criteria_statuses


_password = "<<Your Password>>"

## initialize state
if "results_obj" not in st.session_state:
    st.session_state.results_obj = None
if "human_feedback" not in st.session_state:
    st.session_state.human_feedback = []
if "already_saved" not in st.session_state:
    st.session_state.already_saved = False
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "redis_manager" not in st.session_state:
    st.session_state.redis_manager = None
# if "mrn_input" not in st.session_state:
#     st.session_state.mrn_input = None
# if "protocol_selector" not in st.session_state:
#     st.session_state.protocol_selector = None
if "task_iteration" not in st.session_state:
    st.session_state.task_iteration = None
if "experiment" not in st.session_state:
    st.session_state.experiment = None

## set up redis connection
if len(sys.argv) == 2:
    REDIS_HOST, REDIS_PORT = sys.argv[1].split(":")[:2]
else:
    REDIS_HOST = "localhost"
    REDIS_PORT = 6379

try:
    st.session_state.redis_manager = RedisManager(host=REDIS_HOST, port=REDIS_PORT)
except redis.exceptions.ConnectionError:
    st.error(
        f"Could not connect to Redis server at {REDIS_HOST}:{REDIS_PORT}. Please check your connection."
    )
    st.stop()


### Functions
def check_password():
    """
    Returns `True` if the user had the correct password.
    Based on: https://docs.streamlit.io/knowledge-base/deploy/authentication-without-sso
    But for now we are just comparing with a string, not using secrets. Maybe will add that later to increase security.
    This is not meant to be production grade security at this point.
    """

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == _password:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    # st.title("MSK-Match: AI Clinical Trial Matcher")
    st.text_input(
        "Enter Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error(
            "Password incorrect. For TrialMatcher access, please contact lia5@mskcc.org"
        )
    return False


def load_inputs(
    *, mrn: str, protocol: str, example_mode: bool = False, experiment: str = None
):
    """
    Loads a new results object based on the selected MRN and protocol.
    Also loads the human feedback, if it exists.
    This is called when the user wants to change to a new (patient, protocol) pair.
    """

    if example_mode:
        ai_output = st.session_state.redis_manager.get_example_output()
        human_output = None
    elif experiment:
        ai_output = st.session_state.redis_manager.get_experiment_results(
            experiment_name=experiment, mrn=mrn, protocol=protocol
        )
        human_output = None
    else:
        # loading from redis
        ai_output, human_output = (
            st.session_state.redis_manager.get_most_recent_outputs(mrn, protocol)
        )

    if ai_output:
        results_obj = TrialMatcherState.model_validate_json(ai_output)
        st.session_state.results_obj = results_obj
    else:
        st.error(f"Could not load results for {mrn=}, {protocol=}")
        st.session_state.results_obj = None
        return

    if human_output:
        human_feedback_obj = HumanFeedback.model_validate_json(human_output)
        st.session_state.human_feedback = human_feedback_obj.human_feedback
        st.session_state.already_saved = True
    else:
        st.session_state.human_feedback = []
        st.session_state.already_saved = False

    st.session_state.start_time = time.time()
    return


def load_next_task():
    """
    Load the next unannotated task from Redis.
    This function is called when the user clicks the "Next" button.
    """
    task = st.session_state.redis_manager.get_next_unannotated_task(
        iteration=st.session_state.iteration_selected
    )
    if task is None:
        st.error("No more unannotated tasks available!")
        return
    mrn, protocol, task_iteration = task
    st.session_state.mrn_input = mrn
    st.session_state.protocol_selector = protocol
    st.session_state.task_iteration = task_iteration
    load_inputs(mrn=mrn, protocol=protocol)


def save_outputs(*, mrn, protocol):
    """
    Saves the human feedback.
    Note that the AI results are not saved here, only the human feedback, because the AI results aren't modified themselves.
    """
    if st.session_state.already_saved:
        st.error("You have already saved feedback for this patient.")
        return False

    time_duration = time.time() - st.session_state.start_time

    # put all the human feedback into a single object
    human_data = HumanFeedback(
        trial_id=protocol,
        mrn=mrn,
        human_feedback=st.session_state.human_feedback,
        time_duration=time_duration,
    )

    st.session_state.redis_manager.add_human_output(
        mrn=mrn,
        protocol=protocol,
        result=human_data.model_dump_json(),
    )

    # save the human feedback
    for feedback in st.session_state.human_feedback:
        if feedback.human_explanation:
            st.session_state.redis_manager.add_human_feedback(
                feedback=feedback.human_explanation
            )

    # if we get here, we successfully saved the human feedback
    st.session_state.already_saved = True
    st.success("Feedback saved!")
    # st.rerun()


def save_and_load_next_task(*, mrn, protocol):
    """
    Save the current results and load the next task.
    This is called when the user clicks the "Next" button.
    """
    save_outputs(mrn=mrn, protocol=protocol)
    load_next_task()


def get_updated_results_obj() -> TrialMatcherState:
    """
    Takes the original results object and applies the human feedback to it.
    This should be the way to get the results object to display in the UI- NOT by accessing the session state directly.
    """
    results_original = st.session_state.results_obj
    human_feedback = st.session_state.human_feedback

    out = results_original.copy(deep=True)

    for crit in out.completed_criteria:
        crit.answered_by = "AI"

    # apply the human feedback diffs
    for feedback in human_feedback:
        for crit in out.completed_criteria:
            if crit.id == feedback.criterion_id:
                crit.determination = feedback.human_determination
                crit.answered_by = "human"
                crit.explanation["human feedback"] = feedback.human_explanation
    return out


def setup_criteria_table():
    col_layout = [1, 2, 4, 1, 1]
    col_names = ["ID", "Criterion", "Explanation", "", "Status"]
    cols = st.columns(col_layout)
    for c, v in enumerate(col_names):
        cols[c].subheader(v)

    results = get_updated_results_obj()

    for criterion in results.completed_criteria:
        cols = st.columns(col_layout)

        if f"status_{criterion.id}" in st.session_state:
            criterion.determination = st.session_state[f"status_{criterion.id}"].lower()

        if criterion.determination == "unable to determine":
            criterion_color = "blue"
        elif (
            criterion.criterion_type == "inclusion" and criterion.determination == "met"
        ) or (
            criterion.criterion_type == "exclusion"
            and criterion.determination == "not met"
        ):
            criterion_color = "green"
        else:
            criterion_color = "red"

        cols[0].markdown(f"**:{criterion_color}-background[{criterion.id}]**")
        cols[1].text(criterion.criterion_text)
        if isinstance(criterion.explanation, dict):
            explanation_string = "\n\n".join(
                f"**{expert}:** {exp}" for expert, exp in criterion.explanation.items()
            )
        else:
            explanation_string = criterion.explanation
        cols[2].markdown(explanation_string)
        # Add button to inspect RAG evidence
        cols[3].button(
            "Inspect Evidence",
            on_click=show_rag_dialog,
            kwargs={"criterion": criterion},
            key=f"inspect_{criterion.id}",
            disabled=criterion.rag_docs is None,
        )

        options = ["Met", "Not Met", "Unable to determine"]
        selected_index = [status.lower() for status in options].index(
            criterion.determination.lower()
        )
        cols[4].selectbox(
            "Status",
            options=options,
            key=f"status_{criterion.id}",
            index=selected_index,
            on_change=show_feedback_dialog,
            kwargs={"criterion": criterion},
            disabled=st.session_state.already_saved,
            help="Feedback already saved for this patient"
            if st.session_state.already_saved
            else None,
        )


# function to enter expert input
@st.dialog("Enter Feedback to Improve AI Predictions", width="large")
def enter_feedback(criterion: Criterion):
    if st.session_state.already_saved:
        st.error("You have already saved feedback for this patient.")
    st.write(
        "You chose to override the AI prediction based on your expertise! Please provide feedback that will be used to help improve the AI predictions in the future. Try to give advice that will be generally applicable for future patients."
    )
    st.subheader(criterion.id.capitalize())
    st.text(criterion.criterion_text)
    st.subheader("AI Prediction:")
    if isinstance(criterion.explanation, dict):
        explanation_string = "\n\n".join(
            f"**{expert}:** {exp}" for expert, exp in criterion.explanation.items()
        )
    else:
        explanation_string = criterion.explanation
    st.markdown(explanation_string)
    feedback = st.text_area(
        "Feedback: How can this assessment be improved?", key=f"feedback_{criterion.id}"
    )

    if st.button("Save", type="primary", disabled=st.session_state.example_mode):
        human_feedback = HumanFeedbackSingle(
            criterion_id=criterion.id,
            human_determination=st.session_state[f"status_{criterion.id}"].lower(),
            human_explanation=feedback,
        )
        # print(human_feedback)
        # print(len(st.session_state.human_feedback))
        st.session_state.human_feedback.append(human_feedback)
        # print(len(st.session_state.human_feedback))
        st.success("Feedback saved!", icon="✅")
        # st.rerun()


@st.dialog("Inspect documet snippets used by AI model", width="large")
def inspect_rag_evidence(criterion: Criterion):
    st.write(
        "The AI model used the following snippets from the patient's medical record to make its determination:"
    )
    rag_docs_dict = {
        f"{d.metadata['type']} ({d.metadata['procedure_date']}) [{i}]": re.sub(
            r"\s+", " ", d.page_content.strip()
        )
        for i, d in enumerate(criterion.rag_docs)
    }
    # Dropdown menu
    selected_option = st.pills("Select a Document:", rag_docs_dict.keys())

    if selected_option:
        # Display selected option in markdown
        st.markdown(rag_docs_dict[selected_option])
    else:
        st.info("Select a document to view its content.")


# Create a separate function to toggle the dialog displays
# without this, was running into issues with state
def show_rag_dialog(criterion: Criterion):
    # Track which dialog should be shown
    st.session_state.rag_dialog_to_show = criterion.id
    st.session_state[criterion.id] = criterion


def show_feedback_dialog(criterion: Criterion):
    st.session_state.feedback_dialog_to_show = criterion.id
    st.session_state[criterion.id] = criterion


@st.dialog("Technical Details", width="large")
def show_technical_details():
    info_string
    st.markdown(f"🟢 Connected to Redis server at `{REDIS_HOST}:{REDIS_PORT}`")
    if example_mode:
        st.markdown("Using example data. Redis key: `example_deid_1234`")
        return
    if experiment:
        st.markdown(f"Experiment: `{experiment}`")
        st.markdown(f"Redis key: `experiment:{experiment}:{mrn}_{protocol}_results`")
    else:
        st.markdown(
            f"Redis key: `{mrn}_{protocol}_output_{st.session_state.task_iteration}`"
        )
    tech = "**Run configuration:**"
    for k, v in st.session_state.results_obj.run_config.model_dump().items():
        tech += f"\n- {k}: `{v}`"
    st.markdown(tech)


### Page Configuration
info_string = f"""
**TrialMatcher**  
Version {trialmatcher.__version__}  
Developed by Jacob Rosenthal (RosentJ@mskcc.org) and Anyi Li (LiA5@mskcc.org)
"""

st.set_page_config(
    page_title="MSK-Match: AI Clinical Trial Matcher",
    layout="wide",
    menu_items={"about": info_string},
)

### Main Page Layout
st.title("MSK-Match: AI Clinical Trial Matcher")
st.logo("src/trialmatcher/app/msk_logo.png", size="large")

# Password Check
if not check_password():
    st.stop()  # Do not continue if check_password is not True.

# Protocol and MRN Selection in the Same Line
cols_top = st.columns([1, 2, 2, 1], vertical_alignment="center")

human_review_mode = cols_top[0].toggle(label="Review Mode", key="human_review_mode")
experiment = cols_top[0].selectbox(
    label="Experiment name",
    options=[
        "",
        "base_config",
        # "base_config_no_feedback",
        # "base_config_no_multiexpert",
    ],
    key="experiment",
    index=1,
)
example_mode = cols_top[0].toggle(label="Example Mode", key="example_mode")
protocols = list(all_trial_criteria.keys())

if st.session_state.human_review_mode and not st.session_state.example_mode:
    # Determine the default protocol value (if not set, default to first option)
    default_protocol = st.session_state.protocol_selector or protocols[0]
    default_index = (
        protocols.index(default_protocol) if default_protocol in protocols else 0
    )

    protocol = cols_top[1].selectbox(
        "Select Protocol",
        options=protocols,
        key="protocol_selector",
        # index=default_index,
        disabled=True,
    )
    mrn = cols_top[2].text_input(
        "Enter Patient MRN",
        key="mrn_input",
        value=st.session_state.mrn_input or "",
        disabled=True,
    )

    cols_top[3].button(
        "Next →",
        on_click=load_next_task,
        use_container_width=True,
    )
else:
    protocol = cols_top[1].selectbox(
        "Select Protocol",
        options=protocols,
        key="protocol_selector",
        placeholder="Select a protocol",
    )
    mrn = cols_top[2].text_input(
        "Enter Patient MRN",
        key="mrn_input",
        placeholder="Enter MRN",
    )
    # Submit Button at the Top
    cols_top[3].button(
        "Submit",
        on_click=load_inputs,
        kwargs={
            "mrn": mrn,
            "protocol": protocol,
            "example_mode": example_mode,
            "experiment": experiment,
        },
        use_container_width=True,
    )

if st.session_state.example_mode:
    st.info(
        "Example mode is enabled. Using deidentified example data for testing/demonstration purposes only. No data will be saved."
    )

st.divider()

if st.session_state.results_obj is not None:
    results = get_updated_results_obj()
    # top-level numbers for criteria
    counts = {"qualifying": 0, "disqualifying": 0, "unable to determine": 0}

    # update criteria statuses with selections, if applicable
    for crit in results.completed_criteria:
        if f"status_{crit.id}" in st.session_state:
            crit.determination = st.session_state[f"status_{crit.id}"].lower()

    counts = count_criteria_statuses(results)

    cols_counts = st.columns([1, 1, 1, 2.5, 1], gap="large")
    cols_counts[0].markdown(
        f"Qualifying: :green-background[:green[{counts['qualifying']}]]"
    )
    cols_counts[1].markdown(
        f"Disqualifying: :red-background[:red[{counts['disqualifying']}]]"
    )
    cols_counts[2].markdown(
        f"Unable to Determine: :blue-background[:blue[{counts['unable to determine']}]]"
    )

    # Update Final Determination
    results = final_determination_rule_based(results)

    color_pred = "green" if results.final_determination == "eligible" else "red"
    cols_counts[3].markdown(
        f"**Eligibility Status: :{color_pred}-background[:{color_pred}[{results.final_determination.capitalize()}]]**"
    )
    cols_counts[3].warning(
        "This assessment is made by AI and may contain inaccuracies. It should always be reviewed carefully."
    )
    cols_counts[4].button(
        "Show Technical Details",
        on_click=show_technical_details,
        use_container_width=True,
    )
    st.divider()
    setup_criteria_table()
    # check if the dialog should be shown
    if hasattr(st.session_state, "rag_dialog_to_show"):
        criterion_id = st.session_state.rag_dialog_to_show
        if criterion_id not in st.session_state:
            st.error(f"Error: couldn't find criterion_id {criterion_id}")
            st.stop()
        inspect_rag_evidence(st.session_state[criterion_id])
        del st.session_state.rag_dialog_to_show

    if hasattr(st.session_state, "feedback_dialog_to_show"):
        criterion_id = st.session_state.feedback_dialog_to_show
        if criterion_id not in st.session_state:
            st.error(f"Error: couldn't find criterion_id {criterion_id}")
            st.stop()
        enter_feedback(st.session_state[criterion_id])
        del st.session_state.feedback_dialog_to_show

else:
    st.warning("No results loaded.")

# Save Button
if human_review_mode:
    lab = "Save results and load next patient for review"
    st.button(
        label=lab,
        type="primary",
        disabled=st.session_state.already_saved or st.session_state.results_obj is None,
        on_click=save_and_load_next_task,
        kwargs={
            "mrn": st.session_state.mrn_input,
            "protocol": st.session_state.protocol_selector,
        },
    )
else:
    lab = (
        "Save results"
        if not st.session_state.already_saved
        else "Changes already saved"
    )
    st.button(
        label=lab,
        type="primary",
        disabled=st.session_state.already_saved
        or st.session_state.results_obj is None
        or st.session_state.example_mode
        or experiment is not None,
        on_click=save_outputs,
        kwargs={
            "mrn": st.session_state.mrn_input,
            "protocol": st.session_state.protocol_selector,
        },
    )
