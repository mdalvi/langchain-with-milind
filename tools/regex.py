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


def get_linkedin_username(profile_url: str) -> str | None:
    pattern = r"^https://www\.linkedin\.com/in/([^/]+)/.*"
    match = re.match(pattern, profile_url)
    if match:
        return match.group(1)
    else:
        return None


def get_youtube_video_id(url):
    regex_pattern = r"(?:v=|\/videos\/|embed\/|youtu.be\/|\/v\/|\/e\/|watch\?v=|&v=|\?v=|\/watch\?v=)([a-zA-Z0-9_-]{11})"
    match = re.search(regex_pattern, url)
    if match:
        return match.group(1)
    return None
