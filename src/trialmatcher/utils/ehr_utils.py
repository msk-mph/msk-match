import pandas as pd
from pathlib import Path
import logging


logger = logging.getLogger("trialmatcher")


def process_dumped_ehr_data(mrn, data_dir="data/patient_records"):
    """
    Load and do some basic preprocessing of EHR data.
    Operates on a .json file, e.g. the output of `utils.get_records.get_all_docs()`
    Returns dataframes of structured and unstructured data elements

    :param mrn: Medical record number
    :param data_dir: Path to directory containing .json files of dumped ehr data. Defaults to "data/patient_records"
    :return: Tuple of data_structured, data_unstructured
    """
    data_path = f"{data_dir}/docs_{mrn}.json"

    # if data_path doesn't exist, download the files
    if not Path(data_path).exists():
        raise Exception(f"can't find data at {data_path}")

    data = pd.read_json(data_path)
    logger.debug(f"Loaded {len(data)} records for MRN {mrn}")
    logger.debug(f"Columns: {data.columns}")
    data.drop(["mrn", "id"], axis=1, inplace=True)
    data["procedure_date"] = pd.to_datetime(
        data["procedure_date"], format="mixed", errors="coerce"
    )
    data_unstructured = data[
        data["content"].apply(lambda x: isinstance(x, str))
    ].reset_index(drop=True)
    data_structured = data[
        data["content"].apply(lambda x: isinstance(x, list))
    ].reset_index(drop=True)
    data_unstructured["content"] = data_unstructured["content"].astype(str)
    return data_structured, data_unstructured
