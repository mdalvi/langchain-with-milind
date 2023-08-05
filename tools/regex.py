import re


def get_twitter_username(text: str) -> str:
    """
    Extract pattern starting with `@` in given text and returns the selection.
    :param text: input string to query
    :return: regex selection
    """
    pattern = r"@(\w+)"
    matches = re.findall(pattern, f"{text}")
    if matches:
        return matches[0]
    return ""
