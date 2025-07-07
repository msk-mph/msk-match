import json
import signal
import subprocess
import pandas as pd
import argparse
import logging
from tqdm import tqdm
import traceback

from trialmatcher.utils.schemas import TrialMatcherConfig
from trialmatcher.utils import (
    setup_logging,
    RedisManager,
    AzureClient,
    prep_vector_store,
)
from trialmatcher.langgraph import run_langgraph_trial_matcher


# Define a custom exception for timeout
class TimeoutException(Exception):
    pass


# Timeout handler function
def timeout_handler(signum, frame):
    raise TimeoutException


# Wrapper to call function with a timeout mechanism
def run_with_timeout(func, *, timeout, **kwargs):
    # Set the timeout handler
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)  # Start the countdown
    try:
        result = func(**kwargs)  # Call the function
        signal.alarm(0)  # Cancel the alarm if function completes in time
        return result
    except TimeoutException:
        print(f"Function call timed out after {timeout} seconds.")
        return None


def get_current_git_commit_hash() -> str:
    """Gets the hash for the current Git commit. Uses the short hash for brevity.

    Returns:
        str: short hash of current commit
    """
    try:
        # Run the git rev-parse HEAD command
        commit_hash = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.STDOUT
            )
            .strip()
            .decode("utf-8")
        )
        return commit_hash
    except subprocess.CalledProcessError as e:
        # Handle errors (e.g., if not in a git repository)
        print("Error getting Git commit hash:", e.output.decode("utf-8"))
        return "unknown"


def verify_dataset_columns(df: pd.DataFrame):
    """Make sure the dataset has the required columns.

    Args:
        df (pd.DataFrame): dataframe to verify
    """
    # Required columns
    required_columns = {
        "MRN",
        "protocol",
    }
    # Assert the DataFrame contains the required columns
    assert required_columns.issubset(
        df.columns
    ), f"Missing columns: {required_columns - set(df.columns)}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run experiment with config and dataset files."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config JSON file",
        dest="config_path",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the dataset CSV file. Must have columns: 'MRN' and 'protocol'. Can also provide columns 'eligibility_status' and 'eligibility_status_date' if they exist, i.e. if the patient's ground truth eligibility status as determined on a certain date is known.",
        dest="dataset_path",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        required=False,
        help="Optionally specify epoch number. If an epoch is specified, outputs already present in the database for that epoch won't be re-run (allowing for resuming of interrupted runs) and inference will only be run for elements where outputs for epoch n-1 exist. Example: if epoch=2, only elements where outputs for epoch 1 exist will be run, and any outputs already processed in epoch 2 will be skipped.",
    )
    parser.add_argument(
        "--predownload",
        action="store_true",
        help="If set, pre-download patient records and pre-compute vectorstores for the dataset without running the full experiment.",
    )
    parser.add_argument(
        "--pair",
        nargs=2,
        action="append",
        metavar=("MRN", "PROTOCOL"),
        help="Specify an (MRN, protocol) pair to include in the run. Can be specified multiple times.",
    )
    parser.add_argument(
        "--logfile",
        type=str,
        default="trialmatcher.log",
        help="Name of the log file to write to. Defaults to 'trialmatcher.log'.",
        dest="logfile_name",
    )
    parser.add_argument(
        "--logging_level",
        type=int,
        default=logging.INFO,
        help="Logging level for the run. Defaults to logging.INFO. Use logging.DEBUG for more verbose output.",
        dest="logging_level",
    )
    parser.add_argument(
        "--console_logging_level",
        type=int,
        default=logging.WARNING,
        help="Logging level for console output. Defaults to logging.WARNING. Use logging.DEBUG for more verbose output.",
        dest="console_logging_level",
    )
    return parser.parse_args()


def run_experiment(
    run_config: TrialMatcherConfig,
    dataset: pd.DataFrame,
    epoch: int = None,
    logfile_name: str = "trialmatcher.log",
    logging_level: int = logging.INFO,
    console_logging_level: int = logging.WARNING,
):
    """Run the experiment with the given config and dataset.

    Args:
        run_config (TrialMatcherConfig): configuration for the run
        dataset (pd.DataFrame): dataframe containing MRN and protocol columns
        epoch (int, optional): epoch number. Defaults to None.
    """
    # Verify dataset columns
    verify_dataset_columns(dataset)

    # update git commit
    run_config.git_commit = get_current_git_commit_hash()

    if run_config.debug:
        logging_level = logging.DEBUG
        console_logging_level = logging.DEBUG

    logger = setup_logging(
        log_dir=run_config.output_dir,
        log_file=logfile_name,
        level=logging_level,
        console_level=console_logging_level,
    )

    logger.info("starting run")
    logger.info(f"Run config: {run_config}")

    # initialize AzureClient outside loop to avoid reinitializing for each row
    azure_client = AzureClient(run_config)

    # Initialize RedisManager
    redis_manager = RedisManager(host=run_config.redis_host, port=run_config.redis_port)

    logger.info(f"Dataset shape: {dataset.shape}")

    # Process each row in the dataset
    pbar = tqdm(dataset.iterrows(), total=dataset.shape[0])
    for index, row in pbar:
        mrn = row["MRN"]
        trial_id = row["protocol"]

        pbar.set_description(f"Processing MRN {mrn} | Trial {trial_id}")

        if epoch is not None:
            # skip if already exists
            if redis_manager.client.exists(
                redis_manager._ai_key(mrn=mrn, protocol=trial_id, index=epoch)
            ):
                logger.info(
                    f"Skipping because already exists in redis. MRN {mrn} | Trial {trial_id} | Epoch {epoch}"
                )
                continue
            else:
                logger.info(
                    f"checked that output doesn't already exist in redis for {mrn=}, {trial_id=}, {epoch=}. Continuing."
                )
            # skip if previous epoch doesn't exist
            if epoch != 0:
                if not redis_manager.client.exists(
                    redis_manager._ai_key(mrn=mrn, protocol=trial_id, index=epoch - 1)
                ):
                    logger.info(
                        f"Skipping because previous epoch doesn't exist in redis. MRN {mrn} | Trial {trial_id} | Epoch {epoch}"
                    )
                    continue
                else:
                    logger.info(
                        f"checked that previous epoch exists in redis for {mrn=}, {trial_id=}, {epoch=}. Continuing."
                    )

        if "eligibility_status" in row:
            if row["eligibility_status"] == "Eligible":
                eligibility_ground_truth = "eligible"
            elif row["eligibility_status"] == "Not Eligible":
                eligibility_ground_truth = "ineligible"
            else:
                eligibility_ground_truth = None
        else:
            eligibility_ground_truth = None

        if "eligibility_status_date" in row:
            cutoff_date = row["eligibility_status_date"]
        else:
            cutoff_date = None

        kwargs = {
            "mrn": mrn,
            "trial_id": trial_id,
            "run_config": run_config,
            "azure_client": azure_client,
            "cutoff_date": cutoff_date,
            "current_date": cutoff_date,
            "eligibility_ground_truth": eligibility_ground_truth,
        }

        try:
            run_with_timeout(
                run_langgraph_trial_matcher, timeout=run_config.timeout, **kwargs
            )
        except Exception:
            logger.error(
                f"Error processing MRN {mrn} | Trial {trial_id} | Cutoff {cutoff_date}"
            )
            logger.info(traceback.format_exc())
            continue


def run_predownload(run_config: TrialMatcherConfig, dataset: pd.DataFrame):
    """Pre-download vectorstores for each row in the dataset."""
    # Verify dataset columns
    verify_dataset_columns(dataset)

    logger = setup_logging(
        log_dir=run_config.output_dir,
        log_file="trialmatcher.log",
        level=logging.INFO if not run_config.debug else logging.DEBUG,
        console_level=logging.WARNING if not run_config.debug else logging.DEBUG,
    )
    logger.info("Starting pre-download of vectorstores")

    # Initialize AzureClient
    azure_client = AzureClient(run_config)

    # Compute latest cutoff date per MRN
    if "eligibility_status_date" in dataset.columns:
        latest_dates = dataset.groupby("MRN")["eligibility_status_date"].max()
    else:
        latest_dates = pd.Series({mrn: None for mrn in dataset["MRN"].unique()})

    # Iterate over each MRN and its latest date
    pbar = tqdm(latest_dates.items(), total=len(latest_dates))
    for mrn, cutoff_date in pbar:
        pbar.set_description(f"Pre-download MRN {mrn} | Cutoff {cutoff_date}")
        try:
            vectorstore, vectorstore_tokens = prep_vector_store(
                mrn=mrn,
                embedding_model=azure_client.langchain_azure_openai_embeddings,
                cutoff_date=cutoff_date,
                run_config=run_config,
                disable_tqdm=not run_config.debug,
            )
        except Exception:
            logger.error(f"Error pre-downloading vectorstore for MRN {mrn}")
            logger.error(traceback.format_exc())
            continue


def main():
    """main entry point for CLI"""
    args = parse_args()

    # load config Pydantic model from json
    with open(args.config_path) as f:
        config_data = json.load(f)
    run_config = TrialMatcherConfig.model_validate(config_data)

    # load dataset, verify that it has all the required columns
    df = pd.read_csv(args.dataset_path, dtype={"MRN": str})
    verify_dataset_columns(df)

    # Filter dataset by specified MRN/protocol pairs if provided
    if hasattr(args, "pair") and args.pair:
        pairs = set((mrn, protocol) for mrn, protocol in args.pair)
        # Filter rows matching any of the specified pairs
        df = df[df.apply(lambda row: (row["MRN"], row["protocol"]) in pairs, axis=1)]
        if df.empty:
            print(f"No rows match specified MRN/protocol pairs: {pairs}")
        else:
            print(f"Filtered dataset to specified MRN/protocol pairs: {pairs}")

    if args.predownload:
        run_predownload(
            run_config=run_config,
            dataset=df,
            logfile_name=args.logfile_name,
            logging_level=args.logging_level,
            console_logging_level=args.console_logging_level,
        )
    else:
        run_experiment(
            run_config=run_config,
            dataset=df,
            epoch=args.epoch,
            logfile_name=args.logfile_name,
            logging_level=args.logging_level,
            console_logging_level=args.console_logging_level,
        )


if __name__ == "__main__":
    main()
