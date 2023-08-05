# noinspection PyUnresolvedReferences
import json

# noinspection PyUnresolvedReferences
import joblib
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from pathlib import Path
from agents.linkedin import get_linkedin_profile_url

# noinspection PyUnresolvedReferences
from third_parties.linkedin import get_linkedin_profile, get_saved_linkedin_profile


def run_chain_for_information() -> None:
    """
    Prints short summary of the provided information along with two interesting facts about the person
    :return: None
    """
    with open(Path("storage/wikipedia/donald.trump.txt"), "r") as f:
        information = f.read()

    custom_template = """
    Given the information {information} about a person, create 
    1. a short summary
    2. two interesting facts about the person
    """

    prompt = PromptTemplate(input_variables=["information"], template=custom_template)
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    chain = LLMChain(llm=llm, prompt=prompt)
    print(chain.run(information=information))


def run_chain_for_linkedin_url(name: str) -> str:
    """
    Searches and returns the LinkedIn profile page url of the person
    :param name: Name of person to Google search
    :return: profile page url
    """
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    result = get_linkedin_profile_url(llm, name)
    print(result)
    return result


def run_chain_for_linkedin(name: str) -> None:
    """
    Step 1: Searches google for LinkedIn URL of the person
    Step 2: Fetches public data from LinkedIn about the person
    Step 3: Generates summary and two interesting facts about the person using LLM
    :return: None
    """
    # profile = get_linkedin_profile(
    #     profile_url=run_chain_for_linkedin_url(name)
    # )
    # data = profile.content
    # with open(f"storage/linkedin/{name}.json", "wb") as f:
    #     f.write(data)
    #
    information = get_saved_linkedin_profile("storage/linkedin/milind.dalvi.json")

    custom_template = """
        Given the LinkedIn information {information} about a person, create
        1. a short summary
        2. two interesting facts about the person
        """

    prompt = PromptTemplate(input_variables=["information"], template=custom_template)
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    chain = LLMChain(llm=llm, prompt=prompt)
    print(chain.run(information=information))


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    print("Hello LangChain!")
    run_chain_for_linkedin_url("Ranjan Pradhan Capgemini")
