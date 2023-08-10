import json
import os
import shutil
from typing import Tuple

from langchain import PromptTemplate
from langchain.chains import LLMChain

# noinspection PyUnresolvedReferences
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.output_parsers import PydanticOutputParser

from parsers.pydantic import PersonalIntel
from third_parties.linkedin import get_linkedin_profile, get_saved_linkedin_profile
from tools.regex import get_linkedin_username
from tools.requests import get_image_from_url

GENDER_PREFIX = [
    "Mr.",
    "Dr.",
    "Prof.",
    "Sir",
    "Ms.",
    "Mrs.",
    "Miss",
    "Prof.",
    "Madam",
    "Ma'am",
]


def run_chain_for_linkedin(
    linkedin_url: str, prefix: str
) -> Tuple[PersonalIntel, dict]:
    """
    Step 1: Fetches public data from LinkedIn about the person using the url
    Step 2: Generates summary and two interesting facts about the person using LLM and above information
    :return: None
    """
    user_name = get_linkedin_username(profile_url=linkedin_url)
    file_path = f"storage/linkedin/{user_name}.json"
    if os.path.exists(file_path):
        print("Using saved profile to process request.")
        profile_data = get_saved_linkedin_profile(f"storage/linkedin/{user_name}.json")
    else:
        print("Invoking nubela.co for profile details.")
        profile = get_linkedin_profile(profile_url=linkedin_url)
        with open(f"storage/linkedin/{user_name}.json", "w") as f:
            f.write(json.dumps(profile))
        profile_data = get_saved_linkedin_profile(f"storage/linkedin/{user_name}.json")

    image_path = f"storage/linkedin/images/{user_name}.jpg"
    if not os.path.exists(image_path):
        if not get_image_from_url(
            profile_data["profile_pic_url"], f"storage/linkedin/images/{user_name}.jpg"
        ):
            shutil.copy(
                f"storage/linkedin/images/no_image.jpg",
                image_path,
            )
    profile_data.pop("profile_pic_url", None)

    out_parser = PydanticOutputParser(pydantic_object=PersonalIntel)
    custom_template = """
        Given the LinkedIn information {profile_data} about a person, create
        1. A short summary
        2. Two interesting facts about the person, use gender prefix as {prefix}
        3. A topic that may interest them
        4. Two creative Ice-breakers to open a conversation with them
        \n{format_instructions}
        """

    prompt = PromptTemplate(
        input_variables=["profile_data", "prefix"],
        template=custom_template,
        partial_variables={"format_instructions": out_parser.get_format_instructions()},
    )
    azure_credentials = {
        "temperature": 0,
        "deployment_name": os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
        "openai_api_type": os.environ["AZURE_OPENAI_API_TYPE"],
        "openai_api_base": os.environ["AZURE_OPENAI_API_BASE"],
        "openai_api_version": os.environ["AZURE_OPENAI_API_VERSION"],
        "openai_api_key": os.environ["AZURE_OPENAI_API_KEY"],
    }
    llm = AzureChatOpenAI(**azure_credentials)
    # llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    chain = LLMChain(llm=llm, prompt=prompt)
    return (
        out_parser.parse(chain.run(profile_data=profile_data, prefix=prefix)),
        {
            "image_path": image_path,
            "full_name": profile_data["full_name"],
            "headline": profile_data["headline"],
        },
    )