from typing import List, Union, Literal
from pydantic import BaseModel
from openai import AzureOpenAI
import argparse
import json

from trialmatcher import config


class BaseRecord(BaseModel):
    sub_type: str
    mrn: str
    id: str
    procedure_date: str  # ISO date string


class PathologyRecord(BaseRecord):
    type: Literal["pathology"]
    content: str  # corresponds to report["path_prpt_p1"]


class RadiologyRecord(BaseRecord):
    type: Literal["radiology"]
    content: str  # corresponds to report["RRPT_REPORT_TXT"]


class TestResult(BaseModel):
    test_name: str
    raw_value: Union[str, float]
    text_result: str
    upper_limit: Union[str, float]
    lower_limit: Union[str, float]


class LabRecord(BaseRecord):
    type: Literal["lab"]
    content: List[TestResult]


class ClinicalDocRecord(BaseRecord):
    # type here is the DocTemplateName from Splunk
    content: str  # corresponds to raw_dict["DetailText"]

# Union of all record types
AllRecord = Union[
    PathologyRecord,
    RadiologyRecord,
    LabRecord,
    ClinicalDocRecord,
]

class RecordsResponse(BaseModel):
    records: List[AllRecord]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic patient data and save to a file.")
    parser.add_argument(
        "--output-file", "-o",
        required=True,
        help="Path to write the generated synthetic data JSON."
    )
    args = parser.parse_args()

    azure_client = AzureOpenAI(
        # https://learn.microsoft.com/azure/ai-services/openai/reference#rest-api-versioning
        api_version="2024-08-01-preview",
        azure_endpoint=config.AZURE_OPENAI_API_ENDPOINT,
        api_key=config.AZURE_OPENAI_API_KEY,
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a synthetic data generator that creates realistic patient records "
                "matching the provided Pydantic schema (PathologyRecord, RadiologyRecord, "
                "LabRecord, ClinicalDocRecord). Respond with a JSON array of records, each "
                "conforming to the schema, using realistic values for all fields."
            ),
        },
        {
            "role": "user",
            "content": "Please generate a single synthetic patient record in JSON format for a breast cancer patient meeting all of the following criteria: ECOG performance status 0 or 1; lumpectomy with clear invasive and DCIS margins (no ink on tumor), allowing re-excision if needed (LCIS-only positive margins OK); unilateral invasive breast adenocarcinoma; axillary staging performed; AJCC 8th edition pT1 (≤2 cm) and pN0 (no pN0(i+) or pN0(mol+)); Oncotype DX Recurrence Score ≤18 or MammaPrint 'Low' if Oncotype not available (T1a or international sites exception with block delivery and size ≥0.2 cm; no endocrine therapy prior to tissue collection); ER and/or PgR positive (≥1% by IHC); HER2-negative; pre- or postmenopausal per protocol definition; surgery-to-entry ≤70 days; fully healed incision; bilateral mammogram or MRI within 6 months; HIV+ on effective therapy with undetectable viral load acceptable; intended endocrine therapy for ≥5 years. Make sure that the clinical documents are realistic, spanning the patient's entire disease course over a realistic span of time. Make at least one document in each category, with at least 10 clinical notes. Don't make it obvious that these are synthetic data. The criteria above should be interspersed through the appropriate documents. Each note should be at least 2 pages long, they can use standard medical boilerplate to fill up space"
        },
    ]
    response = azure_client.beta.chat.completions.parse(
        model="gpt-4o-latest",
        messages=messages,
        response_format=RecordsResponse,
        temperature=0.2,
    )
    records = response.choices[0].message.parsed.records
    with open(args.output_file, "w") as out_f:
        json.dump([record.model_dump() for record in records], out_f, indent=2)
    print(f"Synthetic data list written to {args.output_file}")
