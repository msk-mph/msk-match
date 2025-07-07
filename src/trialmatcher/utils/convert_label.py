def convert_label(label: str) -> int:
    """
    Convert eligibility string label to an integer:
      "eligible"   -> 1
      "ineligible" -> 0
    """
    label_lower = label.lower()
    if label_lower == "eligible":
        return 1
    elif label_lower == "ineligible":
        return 0
    else:
        raise ValueError(f"Unexpected label value: {label}")
