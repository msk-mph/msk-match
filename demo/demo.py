# main entry point for demo code

import argparse
import json
import logging
import os

from trialmatcher.langgraph import run_langgraph_trial_matcher
from trialmatcher.utils import AzureClient
from trialmatcher.utils.schemas import TrialMatcherConfig

# Configure logger
logger = logging.getLogger("trialmatcher")
logger.setLevel(logging.INFO)
# Clear any existing handlers
logger.handlers.clear()
# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # set the level you want
console_handler.setFormatter(
    logging.Formatter(
        "MSK-MATCH | %(asctime)s | %(message)s"
    )
)
logger.addHandler(console_handler)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Demonstration code for running MSK-MATCH clinical trial eligibility prescreening."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config JSON file",
        dest="config_path",
    )
    parser.add_argument(
        "--api_endpoint",
        type=str,
        required=True,
        help="API endpoint for Azure OpenAI (url)",
        dest="AZURE_OPENAI_API_ENDPOINT",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        required=True,
        help="API key for Azure OpenAI",
        dest="AZURE_OPENAI_API_KEY",
    )
    parser.add_argument(
        "--mrn",
        type=str,
        required=True,
        help="Patient identifier (MRN)",
        default="1234",
        dest="mrn"
    )
    parser.add_argument(
        "--trial_id",
        type=str,
        required=False,
        help="Clinical trial ID",
        default="16-323",
        dest="trial_id"
    )
    return parser.parse_args()


if __name__ == "__main__":
    logger.info("Starting demo run")
    # main entry point for demo
    args = parse_args()

    try:
        # load run config from json
        with open(args.config_path) as f:
            config_data = json.load(f)
        run_config = TrialMatcherConfig.model_validate(config_data)
        logger.info(f"Successfully loaded config file from {args.config_path}")
    except:
        logger.error(f"Could not load config file from {args.config_path}")
        raise
    
    try:
        # initialize Azure client
        azure_client = AzureClient(
            run_config,
            azure_endpoint=args.AZURE_OPENAI_API_ENDPOINT,
            azure_api_key=args.AZURE_OPENAI_API_KEY,
        )
        # set API key to environment
        os.environ["AZURE_OPENAI_API_KEY"] = args.AZURE_OPENAI_API_KEY
        logger.info("Successfully connected to Azure")
    except:
        logger.error(
            f"Could not connect to Azure with provided arguments:\n\t{args.AZURE_OPENAI_API_ENDPOINT=}\n\t{args.AZURE_OPENAI_API_KEY}"
        )
        raise

    logger.info(f"Running MSK-MATCH for MRN {args.mrn}, trial {args.trial_id}")
    run_langgraph_trial_matcher(
        mrn = args.mrn,
        trial_id=args.trial_id,
        run_config=run_config,
        azure_client=azure_client
    )
    logger.info("Done")