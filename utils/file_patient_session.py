import re


def filename_to_patient_id(filename):
    pattern = r'(\w{8})_rest_'
    match = re.search(pattern, filename)
    return match.group(1)

def filename_to_session(filename):
    pattern = r'_([^_]*)_T(\d\d?)'
    match = re.search(pattern, filename)
    return match.group(2)
