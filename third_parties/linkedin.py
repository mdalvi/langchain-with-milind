import json
import os

import requests
from requests import Response

LINKEDIN_API_ENDPOINT = "https://nubela.co/proxycurl/api/v2/linkedin"


def get_linkedin_profile(profile_url: str) -> Response:
    headers = {"Authorization": f"Bearer {os.environ.get('PROXYCURL_API_KEY')}"}
    response = requests.get(
        LINKEDIN_API_ENDPOINT, params={"url": profile_url}, headers=headers
    )
    return response


def get_saved_linkedin_profile(profile_path: str) -> dict:
    with open(profile_path, "r") as file:
        data = json.load(file)

    data = {
        k: v
        for k, v in data.items()
        if v not in ([], "", None) and k not in ["people_also_viewed", "certifications"]
    }
    if data.get("groups"):
        for dict_ in data["groups"]:
            dict_.pop("profile_pic_url", None)
    data.pop("similarly_named_profiles", None)
    if data.get("experiences"):
        for dict_ in data["experiences"]:
            dict_.pop("logo_url", None)
    return data
