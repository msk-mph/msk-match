from ._21_283 import criteria_21_283
from ._19_300 import criteria_19_300
from ._16_323 import criteria_16_323
from ._18_486 import criteria_18_486
from ._22_259 import criteria_22_259
from ._19_410 import criteria_19_410
from ..utils import Criterion

__all__ = ["all_trial_criteria"]

_trial_criteria = {
    "21-283": criteria_21_283,
    "19-300": criteria_19_300,
    "16-323": criteria_16_323,
    "18-486": criteria_18_486,
    "22-259": criteria_22_259,
    "19-410": criteria_19_410,
}


# update criteria that are vacuous or always require human review
# criteria text : trial id
# these curated by LLM: `12_vacuous_humanreview_criteria.py` and then double checked by human
vacuous_criteria = {
    "The trial is open to female and male patients.": "21-283",
    "For patients who have undergone lumpectomy, any type of mastectomy and any type of reconstruction (including no reconstruction) are allowed. Metallic components of some tissue expanders may complicate delivery of proton therapy; any concerns should be discussed with the Breast Committee Study Chairs prior to registration.": "16-323",
    "For patients who have undergone lumpectomy, there are no breast size limitations.": "16-323",
    "Bilateral breast cancer is permitted. Patients receiving treatment to both breasts for bilateral breast cancer will be stratified as left-sided.": "16-323",
    "Patients may or may not have had adjuvant chemotherapy.": "19-410",
    "Patients with T3N0 disease are eligible.": "19-410",
}

requires_human_review = {
    "The patient or a legally authorized representative must provide study-specific informed consent prior to pre entry /step 1 and, for patients treated in the U.S., authorization permitting release of personal health information": "21-283",
    "Patients must be intending to take endocrine therapy for a minimum 5 years duration (tamoxifen or aromatase inhibitor). The specific regimen of endocrine therapy is at the treating physician's discretion.": "21-283",
    "Written informed consent obtained from subject and ability to comply with the requirements of the study": "19-300",
    "For female subjects of childbearing potential, patient is willing to use 2 methods of birth control or be surgically sterile or abstain from heterosexual activity for the duration of study participation. Note: Should a woman become pregnant while participating on study, she should inform the treating physician immediately": "19-300",
    "Confirmation that the patient's health insurance will pay for the treatment in this study (patients may still be responsible for some costs, such as co-pays and deductibles). If the patient's insurance will not cover a specific treatment in this study and the patient still wants to participate, confirmation that the patient would be responsible for paying for any treatment received.": "16-323",
    "The patient must provide study-specific informed consent prior to study entry.": "16-323",
    "Able to provide informed consent.": "18-486",
    "Patients whose entry to the trial will cause unacceptable clinical delays in their planned management.": "18-486",
    "Willing and able to provide informed consent": "22-259",
    "Consented to 12-245": "22-259",
    "Patient consent must be appropriately obtained in accordance with applicable loca`l and regulatory requirements. Each patient must sign a consent form prior to enrollment in the trial to document their willingness to participate. A similar process must be followed for sites outside of Canada as per their respective cooperative group's procedures.": "19-410",
    "Patients must be accessible for treatment and follow-up. Investigators must assure themselves that patients randomized on this trial will be available for complete documentation of the treatment, adverse events, and follow-up.": "19-410",
    "Patients must have had endocrine therapy initiated or planned for ≥ 5 years. Premenopausal women will receive ovarian ablation plus aromatase inhibitor therapy or tamoxifen if adjuvant chemotherapy was not administered. For all patients, endocrine therapy can be given concurrently or following RT.": "19-410",
    "Has the patient seen their Medical Oncologist?": "19-410",
    "Women of childbearing potential must have agreed to use an effective contraceptive method. A woman is considered to be of 'childbearing potential' if she has had menses at any time in the preceding 12 consecutive months. In addition to routine contraceptive methods, 'effective contraception' also includes heterosexual celibacy and surgery intended to prevent pregnancy (or with a side-effect of pregnancy prevention) defined as a hysterectomy, bilateral oophorectomy or bilateral tubal ligation, or vasectomy/vasectomized partner. However, if at any point a previously celibate patient chooses to become heterosexually active during the time period for use of contraceptive measures outlined in the protocol, she is responsible for beginning contraceptive measures. Women of childbearing potential will have a pregnancy test to determine eligibility as part of the Pre-Study Evaluation (see Section 4.0); this may include an ultrasound to rule-out pregnancy if a false-positive is suspected. For example, when beta-human chorionic gonadotropin is high and partner is vasectomized, it may be associated with tumour production of hCG, as seen with some cancers. Patient will be considered eligible if an ultrasound is negative for pregnancy.": "19-410",
}

all_trial_criteria = {}

for trialid, trialcrit in _trial_criteria.items():
    inc = []
    for c, inc_crit in enumerate(trialcrit["inclusion"], start=1):
        crit = Criterion(
            id=f"inclusion criterion {c}",
            criterion_text=inc_crit,
            criterion_type="inclusion",
            determination=None,
            explanation=None,
        )
        if inc_crit in vacuous_criteria:
            crit.vacuous = True
        if inc_crit in requires_human_review:
            crit.requires_human_review = True
        inc.append(crit)
    exc = []
    for c, exc_crit in enumerate(trialcrit["exclusion"], start=1):
        crit = Criterion(
            id=f"exclusion criterion {c}",
            criterion_text=exc_crit,
            criterion_type="exclusion",
            determination=None,
            explanation=None,
        )
        if exc_crit in vacuous_criteria:
            crit.vacuous = True
        if exc_crit in requires_human_review:
            crit.requires_human_review = True
        exc.append(crit)
    all_trial_criteria[trialid] = inc + exc
