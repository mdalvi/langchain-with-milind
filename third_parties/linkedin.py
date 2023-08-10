import asyncio
import json
import os
from typing import Awaitable

from proxycurl_py.asyncio import Proxycurl
from proxycurl_py.models import PersonEndpointResponse
from pathlib import Path


def get_linkedin_profile(profile_url: str) -> Awaitable[PersonEndpointResponse]:
    proxycurl = Proxycurl(api_key=os.environ["PROXYCURL_API_KEY"])
    person = asyncio.run(
        proxycurl.linkedin.person.get(url=profile_url, fallback_to_cache="on-error")
    )
    return person


def remove_empty_values(d: dict) -> dict:
    if isinstance(d, dict):
        keys_to_remove = [key for key, value in d.items() if value in ([], None, "")]
        for key in keys_to_remove:
            del d[key]
        for value in d.values():
            remove_empty_values(value)
    elif isinstance(d, list):
        for item in d:
            remove_empty_values(item)
    return d


def get_saved_linkedin_profile(profile_path: str) -> dict:
    p = Path(profile_path)
    min_path = f"{p.parent}/min_{p.name}"
    if os.path.exists(min_path):
        with open(min_path, "r") as file:
            data = json.load(file)
    else:
        with open(profile_path, "r") as file:
            data = json.load(file)

        data = remove_empty_values(data)
        data.pop("people_also_viewed", None)
        data.pop("similarly_named_profiles", None)
        data.pop("background_cover_image_url", None)
        data.pop("recommendations", None)
        data.pop("activities", None)
        if data.get("groups"):
            for dict_ in data["groups"]:
                dict_.pop("profile_pic_url", None)
                dict_.pop("url", None)

        if data.get("experiences"):
            for dict_ in data["experiences"]:
                dict_.pop("logo_url", None)
                dict_.pop("company_linkedin_profile_url", None)

        if data.get("certifications"):
            for dict_ in data["certifications"]:
                dict_.pop("url", None)
                dict_.pop("display_source", None)

        with open(min_path, "w") as f:
            f.write(json.dumps(data))
    return data
