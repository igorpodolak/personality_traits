import re


def filename_to_patient_id(filename: str) -> str:
    pattern = r'(\w{8})_rest_'
    match = re.search(pattern, filename)
    return match.group(1)


def filename_to_session(filename: str) -> int:
    pattern = r'_([^_]*)_T(\d\d?)'
    match = re.search(pattern, filename)
    return int(match.group(2))
