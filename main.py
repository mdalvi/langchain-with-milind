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
from third_parties.twitter import scrape_user_tweets
from agents.twitter import get_twitter_profile_username

# noinspection PyUnresolvedReferences
from third_parties.linkedin import get_linkedin_profile, get_saved_linkedin_profile

from langchain.output_parsers import PydanticOutputParser
from parsers.pydantic import PersonalIntel


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
    2. two interesting questions to ask him in job interview
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


def run_chain_for_twitter_username(name: str) -> str:
    """
    Searches and returns the Twitter profile username of the person
    :param name: Name of person to Google search
    :return: Twitter username
    """
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    result = get_twitter_profile_username(llm, name)
    return result


def run_chain_for_social_media(name: str) -> PersonalIntel:
    """
    Step 1: Searches google for LinkedIn URL of the person
    Step 2: Fetches public data from LinkedIn about the person using the url
    Step 3: Searches google for Twitter URL of the person
    Step 4: Scrapes latest tweets about the person from Twitter API
    Step 5: Generates summary and two interesting facts about the person using LLM and above information
    :return: None
    """
    # profile = get_linkedin_profile(profile_url=run_chain_for_linkedin_url(name))
    # data = json.loads(profile.content.decode('utf-8'))
    # pi = data['public_identifier']
    # with open(f"storage/linkedin/{pi}.json", "wb") as f:
    #     f.write(profile.content)

    linkedin_profile = get_saved_linkedin_profile(
        f"storage/linkedin/ranjan.pradhan.json"
    )
    # tweets = scrape_user_tweets(username=run_chain_for_twitter_username(name), num_tweets=5)
    out_parser = PydanticOutputParser(pydantic_object=PersonalIntel)
    custom_template = """
        Given the LinkedIn information {linkedin_profile} about a person, create
        1. A short summary
        2. Two interesting facts about the person
        3. A topic that may interest them
        4. Two creative Ice-breakers to open a conversation with them
        \n{format_instructions}
        """

    prompt = PromptTemplate(
        input_variables=["linkedin_profile"],
        template=custom_template,
        partial_variables={"format_instructions": out_parser.get_format_instructions()},
    )
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    chain = LLMChain(llm=llm, prompt=prompt)
    return out_parser.parse(chain.run(linkedin_profile=linkedin_profile))


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    print("Hello LangChain!")
    result = run_chain_for_social_media("Ranjan Pradhan, Capgemini India")
    print(result.summary, end="\n")
    print(result.facts, end="\n")
    print(result.topics_of_interest, end="\n")
    print(result.ice_breakers)
    # print(scrape_user_tweets(username="@elonmusk", num_tweets=100))
