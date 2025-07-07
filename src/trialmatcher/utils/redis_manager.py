import redis
from typing import Generator, Optional, Tuple, List
import logging
import pandas as pd
import json
from tqdm import tqdm
import copy
import argparse

from trialmatcher.utils.schemas import TrialMatcherState
from trialmatcher.utils.count_criteria_statuses import count_criteria_statuses
from trialmatcher.utils.convert_label import convert_label


logger = logging.getLogger("trialmatcher")


class RedisManager:
    def __init__(self, host: str = "localhost", port: int = 6379) -> None:
        self.host = host
        self.port = port
        try:
            # decode_responses=True ensures we're working with Python strings
            self.client = redis.Redis(host=host, port=port, decode_responses=True)
            self.client.ping()  # Check if the connection is successful
            logger.info(f"Connected to Redis server at {host}:{port}")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis server at {host}:{port}: {e}")
            raise e

    def _master_key(self, mrn: str, protocol: str) -> str:
        """
        Generate the master key for a given MRN and protocol.
        prefix with "master:" to avoid collisions.
        """
        return f"master:{mrn}_{protocol}"

    def _ai_key(self, mrn: str, protocol: str, index: int) -> str:
        """Generate the AI output key for a given index."""
        return f"{mrn}_{protocol}_output_{index}"

    def _human_key(self, mrn: str, protocol: str, index: int) -> str:
        """Generate the human output key for a given index."""
        return f"{mrn}_{protocol}_human_{index}"

    def _initialize_master_key(self, master_key: str) -> None:
        """
        Initialize the master key hash if it does not exist.
        We set initial counts to -1 so that the first addition increments to 0.
        """
        if not self.client.exists(master_key):
            self.client.hset(master_key, mapping={"ai_count": -1, "human_count": -1})

    def add_ai_output(self, mrn: str, protocol: str, result: str) -> int:
        """
        Add an AI output for a given (mrn, protocol) pair.
        Returns the index at which the output was stored.
        """
        master_key = self._master_key(mrn, protocol)
        self._initialize_master_key(master_key)
        # Increment the ai_count field by 1. For the first addition, -1 becomes 0.
        index: int = int(self.client.hincrby(master_key, "ai_count", 1))
        output_key: str = self._ai_key(mrn, protocol, index)
        self.client.set(output_key, result)
        return index

    def add_human_output(self, mrn: str, protocol: str, result: str) -> int:
        """
        Add a human output for a given (mrn, protocol) pair.
        Returns the index at which the output was stored.
        """
        master_key = self._master_key(mrn, protocol)
        self._initialize_master_key(master_key)
        index: int = int(self.client.hincrby(master_key, "human_count", 1))
        output_key: str = self._human_key(mrn, protocol, index)
        self.client.set(output_key, result)
        return index

    def get_most_recent_outputs(
        self, mrn: str, protocol: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Retrieve the most recent AI output, and the corresponding human feedback for that output
        """
        master_key = self._master_key(mrn, protocol)
        if not self.client.exists(master_key):
            return None, None
        ai_count = int(self.client.hget(master_key, "ai_count"))
        if ai_count < 0:
            return None, None

        ai_output_key = self._ai_key(mrn, protocol, ai_count)
        ai_output = self.client.get(ai_output_key)

        human_output_key = self._human_key(mrn, protocol, ai_count)
        human_output = self.client.get(human_output_key)

        return ai_output, human_output

    def get_latest_ai_output(self, mrn: str, protocol: str) -> Optional[str]:
        """
        Retrieve the most recent AI output for a given (mrn, protocol) pair.
        Returns None if no output exists.
        """
        master_key = self._master_key(mrn, protocol)
        if not self.client.exists(master_key):
            return None
        ai_count: int = int(self.client.hget(master_key, "ai_count"))
        if ai_count < 0:
            return None
        output_key: str = self._ai_key(mrn, protocol, ai_count)
        return self.client.get(output_key)

    def get_latest_human_output(self, mrn: str, protocol: str) -> Optional[str]:
        """
        Retrieve the most recent human output for a given (mrn, protocol) pair.
        Returns None if no output exists.
        """
        master_key = self._master_key(mrn, protocol)
        if not self.client.exists(master_key):
            return None
        human_count: int = int(self.client.hget(master_key, "human_count"))
        if human_count < 0:
            return None
        output_key: str = self._human_key(mrn, protocol, human_count)
        return self.client.get(output_key)

    def get_example_output(self) -> str:
        """
        Load deidentified example data for test/demonstration purposes
        """
        return self.client.get("example_deid_1234")

    def _unannotated_task_generator(
        self, incorrect_only: bool, iteration: Optional[int] = None
    ) -> Generator[Tuple[str, str, int], None, None]:
        """
        A generator that yields unannotated tasks as (mrn, protocol, ai_count).

        Args:
            incorrect_only (bool, optional): Whether to only count incorrects as unannotated. Defaults to True.
            iteration (int, optional): The iteration to check for unannotated tasks. Defaults to None.
        """
        master_keys = self.client.keys("master:*")
        for master_key in master_keys:
            ai_count_str = self.client.hget(master_key, "ai_count")
            human_count_str = self.client.hget(master_key, "human_count")
            if ai_count_str is None or human_count_str is None:
                continue  # Skip malformed master keys.
            ai_count = int(ai_count_str)
            human_count = int(human_count_str)

            # Check if there's an unannotated AI output.
            if ai_count > human_count:
                suffix = master_key[len("master:") :]
                if "_" not in suffix:
                    continue
                mrn, protocol = suffix.split("_", 1)
                for task_epoch in range(human_count + 1, ai_count + 1):
                    # skip if it's not the epoch of interest
                    if iteration is not None and task_epoch != iteration:
                        continue
                    # Skip if human output already exists for this epoch:
                    if (
                        self.client.get(self._human_key(mrn, protocol, task_epoch))
                        is not None
                    ):
                        continue
                    if incorrect_only:
                        output_key = self._ai_key(mrn, protocol, task_epoch)
                        output = TrialMatcherState.model_validate_json(
                            self.client.get(output_key)
                        )
                        if (
                            output.final_determination is None
                            or output.eligibility_ground_truth is None
                        ):
                            continue
                        if (
                            output.final_determination
                            == output.eligibility_ground_truth
                        ):
                            continue
                    yield mrn, protocol, task_epoch

    def unannotated_tasks(
        self, incorrect_only: bool = True, iteration: Optional[int] = None
    ) -> List[Tuple[str, str, int]]:
        """Populates a list of unannotated tasks, in the format (mrn, protocol, ai_count).

        Args:
            incorrect_only (bool, optional): Whether to only count incorrects as unannotated. Defaults to True.
            iteration (int, optional): The iteration to check for unannotated tasks. Defaults to None.

        Returns:
            List[Tuple[str, str, int]]: details for unannotated tasks
        """
        return list(self._unannotated_task_generator(incorrect_only, iteration))

    def get_next_unannotated_task(
        self, incorrect_only: bool = True, iteration: Optional[int] = None
    ) -> Optional[Tuple[str, str, int]]:
        """
        Search all master keys and return the first (mrn, protocol) pair
        where there is an unannotated AI output.

        For each master key (of the form "master:{mrn}_{protocol}"), if the AI output count
        is greater than the human feedback count, it means that there is at least one AI output
        that hasn't been annotated.

        If incorrect_only is True, then also ensure that the attributes "final_determination" and
        "eligibility_ground_truth" exist and are not the same (indicating an error).

        The method returns the corresponding (mrn, protocol, task_iteration) tuple.

        If all outputs have been annotated, returns None.
        """
        for mrn, protocol, task_iteration in self._unannotated_task_generator(
            incorrect_only, iteration
        ):
            return mrn, protocol, task_iteration
        return None

    def add_human_feedback(self, feedback: str) -> None:
        """
        Append a new human feedback string to the list stored under the key "human_feedback".
        """
        self.client.rpush("human_feedback", feedback)

    def get_human_feedback(self) -> List[str]:
        """
        Retrieve all human feedback strings from the list stored under the key "human_feedback".
        """
        return self.client.lrange("human_feedback", 0, -1)

    def get_all_ai_outputs(self, mrn: str, protocol: str) -> List[str]:
        """
        Retrieve all AI outputs for a given (mrn, protocol) pair.
        Returns a list of outputs in sequential order.
        """
        outputs: List[str] = []
        master_key = self._master_key(mrn, protocol)
        if not self.client.exists(master_key):
            return outputs
        ai_count = int(self.client.hget(master_key, "ai_count"))
        for i in range(ai_count + 1):
            output = self.client.get(self._ai_key(mrn, protocol, i))
            if output is not None:
                outputs.append(output)
        return outputs

    def get_all_human_outputs(self, mrn: str, protocol: str) -> List[str]:
        """
        Retrieve all human outputs for a given (mrn, protocol) pair.
        Returns a list of human feedback in sequential order.
        """
        outputs: List[str] = []
        master_key = self._master_key(mrn, protocol)
        if not self.client.exists(master_key):
            return outputs
        human_count = int(self.client.hget(master_key, "human_count"))
        for i in range(human_count + 1):
            output = self.client.get(self._human_key(mrn, protocol, i))
            if output is not None:
                outputs.append(output)
        return outputs

    def add_experiment_result(
        self, experiment_name: str, mrn: str, protocol: str, result: str
    ) -> None:
        """
        Append a new experiment result for a given (experiment_name, mrn, protocol) combination.
        The result is stored in a list under the key:
          "experiment:{experiment_name}:{mrn}_{protocol}_results"
        """
        key = f"experiment:{experiment_name}:{mrn}_{protocol}_results"
        self.client.rpush(key, result)

    def get_experiment_results(
        self, experiment_name: str, mrn: str, protocol: str
    ) -> List[str]:
        """
        Retrieve experiment results for a given (experiment_name, mrn, protocol) combination.
        Returns the first result in that list.
        """
        key = f"experiment:{experiment_name}:{mrn}_{protocol}_results"
        return self.client.lrange(key, 0, 0)[0]

    def get_all_results_for_experiment(self, experiment_name: str) -> List[str]:
        """
        Retrieve all experiment results for the given experiment name across all MRNs and protocols.
        This method aggregates the results stored under all keys that match the pattern:
        "experiment:{experiment_name}:{mrn}_{protocol}_results".

        Returns:
            A list of all results from the experiment.
        """
        results: List[str] = []
        pattern = f"experiment:{experiment_name}:*_results"
        # Retrieve all keys matching the experiment pattern.
        keys = self.client.keys(pattern)
        for key in keys:
            # Extend the results list with all entries from the current key's list.
            results.extend(self.client.lrange(key, 0, -1))
        return results

    @staticmethod
    def process_ai_result_for_csv(ai_result_str: str) -> dict:
        # first process the AI predictions
        ai_result = json.loads(ai_result_str)

        mrn = ai_result.get("mrn", None)
        protocol = ai_result.get("trial_id", None)

        final_det_ai = ai_result.get("final_determination", None)
        ground_truth = ai_result.get("eligibility_ground_truth", None)
        if final_det_ai is not None:
            eligibility_pred_ai = convert_label(final_det_ai)
        if ground_truth is not None:
            eligibility_groundtruth = convert_label(ground_truth)

        crit_status_counts = count_criteria_statuses(ai_result_str)

        out = {
            "mrn": mrn,
            "protocol": protocol,
            "eligibility_pred": eligibility_pred_ai,
            "eligibility_groundtruth": eligibility_groundtruth,
            "source": "AI",
            "time_elapsed": ai_result.get("time_elapsed_seconds", None),
            "n_qualifying_criteria": crit_status_counts["qualifying"],
            "n_unable_to_determine_criteria": crit_status_counts["unable to determine"],
            "n_disqualifying_criteria": crit_status_counts["disqualifying"],
            "input_tokens": ai_result.get("input_tokens", None),
            "output_tokens": ai_result.get("output_tokens", None),
            "embedding_tokens": ai_result.get("embedding_tokens", None),
            "cost": ai_result.get("cost", None),
        }

        return out

    def get_all_results_df(self, experiment_name=None) -> pd.DataFrame:
        """
        Retrieve all AI and human results for all MRNs and protocols across all epochs,
        and return them as a pandas DataFrame with the following columns:
            - mrn: The patient MRN.
            - protocol: The protocol identifier.
            - epoch: The index representing the epoch or version of the result.
            - eligibility_pred: Binary eligibility prediction derived from the AI output's
              'final_determination' field, converted using the convert_label helper function.
            - eligibility_groundtruth: Binary eligibility ground truth derived from the human output's
              'eligibility_ground_truth' field, converted using the convert_label helper function.
            - time_elapsed: The time taken for the AI to process the MRN and protocol.

        if experiment_name is provided, it will filter the results to only include those
        associated with the specified experiment. Otherwise, it will include all results from the "master" keys.

        The conversion from string labels ("eligible", "ineligible") to integers (1, 0) is handled by
        the convert_label function. If an error occurs during processing a result (for instance, if the
        result fails validation), the error is logged and the corresponding eligibility value is set to None.

        Returns:
            pd.DataFrame: A DataFrame containing all aggregated results across all MRNs, protocols,
                          and epochs.
        """
        rows = []

        if experiment_name:
            exp_results = self.get_all_results_for_experiment(experiment_name)
            for result_str in tqdm(exp_results):
                new_row = self.process_ai_result_for_csv(result_str)
                rows.append(new_row)

        else:
            master_keys = self.client.keys("master:*")
            for master_key in tqdm(master_keys):
                # Expecting master_key format "master:{mrn}_{protocol}"
                suffix = master_key[len("master:") :]
                if "_" not in suffix:
                    continue
                mrn, protocol = suffix.split("_", 1)
                # Retrieve counts (defaulting to -1 if not found)
                ai_count = int(self.client.hget(master_key, "ai_count") or -1)
                human_count = int(self.client.hget(master_key, "human_count") or -1)
                max_epoch = max(ai_count, human_count)
                if max_epoch < 0:
                    continue
                for epoch in range(max_epoch + 1):
                    ai_result_str = self.client.get(self._ai_key(mrn, protocol, epoch))
                    human_result_str = self.client.get(
                        self._human_key(mrn, protocol, epoch)
                    )

                    if ai_result_str is None:
                        continue

                    processed_ai_result = self.process_ai_result_for_csv(ai_result_str)
                    processed_ai_result["epoch"] = epoch
                    rows.append(processed_ai_result)

                    # next, process human feedback to get AI+human
                    if human_result_str:
                        # start with a copy of the AI result
                        ai_result_copy = copy.deepcopy(ai_result_str)

                        # now apply the human feedback as a diff
                        human_result = json.loads(human_result_str)
                        for f in human_result.get("human_feedback", []):
                            for crit in ai_result_copy.get("completed_criteria", []):
                                if crit["id"] == f["criterion_id"]:
                                    crit["determination"] = f["human_determination"]

                        # now check if the rules-based final_determination has changed
                        # use rule-based logic to make final determination
                        # check that no eligibility criteria are unmet and no exclusion criteria are met
                        final_det_human_ai = "eligible"
                        for crit in ai_result_copy.get("completed_criteria", []):
                            if crit["determination"] == "unable to determine":
                                continue
                            if (
                                crit["criterion_type"] == "inclusion"
                                and crit["determination"] == "not met"
                            ):
                                final_det_human_ai = "ineligible"
                                break
                            if (
                                crit["criterion_type"] == "exclusion"
                                and crit["determination"] == "met"
                            ):
                                final_det_human_ai = "ineligible"
                                break

                        # convert to binary
                        eligibility_pred_human_ai = self._convert_label(
                            final_det_human_ai
                        )

                        # update
                        ai_result_copy_processed = copy.deepcopy(processed_ai_result)
                        ai_result_copy_processed["eligibility_pred"] = (
                            eligibility_pred_human_ai
                        )
                        ai_result_copy_processed["source"] = "AI_human"
                        rows.append(ai_result_copy_processed)

        return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RedisManager CLI: utility for dumping redis data to CSV"
    )
    parser.add_argument("--host", type=str, help="Redis host")
    parser.add_argument("--port", type=int, help="Redis port")
    parser.add_argument(
        "--output", type=str, required=True, help="Output file path for the CSV"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Experiment name to filter results (optional)",
    )
    args = parser.parse_args()

    # Initialize RedisManager
    redis_manager = RedisManager(host=args.host, port=args.port)

    # Get all results as a DataFrame
    results_df = redis_manager.get_all_results_df(experiment_name=args.experiment_name)

    # Save the DataFrame to a CSV file
    results_df.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")
