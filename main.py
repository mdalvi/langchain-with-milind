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


def run_chain_for_information():
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


def run_chain_for_linkedin():
    # profile = get_linkedin_profile(
    #     profile_url="https://www.linkedin.com/in/milinddalvi/"
    # )
    # data = profile.content
    # with open("storage/linkedin/milind.dalvi.json", "wb") as f:
    #     f.write(data)
    #
    information = get_saved_linkedin_profile("storage/linkedin/milind.dalvi.json")

    custom_template = """
        Given the LinkedIn information {information} about a person, create
        1. a short summary
        2. two interesting questions to ask in interview to this person. 
        """
    # facts about the person
    prompt = PromptTemplate(input_variables=["information"], template=custom_template)
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    chain = LLMChain(llm=llm, prompt=prompt)
    print(chain.run(information=information))


def run_chain_for_linkedin_url():
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    print(get_linkedin_profile_url(llm, "Ranjan Pradhan Capgemini"))


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    print("Hello LangChain!")
    run_chain_for_linkedin_url()
