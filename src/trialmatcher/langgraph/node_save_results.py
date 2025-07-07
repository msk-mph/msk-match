"""node to save results to file"""

import logging
import os
from datetime import datetime

from trialmatcher.utils import RedisManager
from trialmatcher.utils.schemas import TrialMatcherState

logger = logging.getLogger("trialmatcher")


def save_results(state: TrialMatcherState):
    logger.info("Saving results")
    # calculate cost of tokens
    # https://openai.com/api/pricing/
    state.cost = round(
        (
            2.5 * state.input_tokens
            + 10 * state.output_tokens
            + 0.13 * state.embedding_tokens
        )
        / 1e6,
        2,
    )
    logger.info(f"Cost: ${state.cost}")
    logger.info(
        f"\tinput tokens: {state.input_tokens}\tcost: {2.5 * state.input_tokens/1e6:.2f}"
    )
    logger.info(
        f"\toutput tokens: {state.output_tokens}\tcost: {10 * state.output_tokens/1e6:.2f}"
    )
    logger.info(
        f"\tembed tokens: {state.embedding_tokens}\tcost: {0.13 * state.embedding_tokens/1e6:.2f}"
    )
    # criteria might be out of order, e.g. because of handling vacuous and human review criteria first.
    # sort the criteria back into correct order before saving, first by type, then by id
    # make sure that inclusion criteria are listed before exclusion criteria
    state.completed_criteria = sorted(
        state.completed_criteria,
        key=lambda x: (
            -1 if x.criterion_type == "inclusion" else 0,
            int(
                x.id.split()[-1]
            ),  # Split by spaces and convert the last element to an integer
        ),
    )

    # track elapsed time
    state.timestamp_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # calculate time elapsed
    if state.timestamp_start and state.timestamp_end:
        start_time = datetime.strptime(state.timestamp_start, "%Y-%m-%d %H:%M:%S")
        end_time = datetime.strptime(state.timestamp_end, "%Y-%m-%d %H:%M:%S")
        state.time_elapsed_seconds = (end_time - start_time).total_seconds()
        logger.info(f"Time elapsed: {state.time_elapsed_seconds} seconds")

    state_json_data = state.model_dump_json(indent=2)

    if state.run_config.redis_host and state.run_config.redis_port:
        # save the results to redis
        redis_manager = RedisManager(
            host=state.run_config.redis_host, port=state.run_config.redis_port
        )
        if state.run_config.experiment_name:
            # in this case, the run is part of an experiment
            redis_manager.add_experiment_result(
                experiment_name=state.run_config.experiment_name,
                mrn=state.mrn,
                protocol=state.trial_id,
                result=state_json_data,
            )
            logger.info(
                f"added experiment result for {state.run_config.experiment_name}"
            )
        else:
            # otherwise, the run is standalone and assumed to be production
            redis_manager.add_ai_output(
                mrn=state.mrn, protocol=state.trial_id, result=state_json_data
            )
        logger.info(
            f"Data saved to redis for MRN {state.mrn} and trial {state.trial_id}"
        )
    else:
        # in this case, no redis -- save results to json file
        out_path = (
            state.run_config.output_dir + f"/{state.mrn}_{state.trial_id}_output.json"
        )
        # Ensure the directory exists
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # Save JSON to a file
        with open(out_path, "w") as f:
            f.write(state_json_data)
        logger.info(f"Data saved to {out_path}")
        logger.info(f"Data saved to file for {state.mrn} and trial {state.trial_id}")

    logger.info(f"Final determination: {state.final_determination}")

    return
