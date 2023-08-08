# noinspection PyUnresolvedReferences
import json
import os
from pathlib import Path

# noinspection PyUnresolvedReferences
import joblib
import pinecone
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.output_parsers import PydanticOutputParser
from langchain.text_splitter import CharacterTextSplitter
import faiss

# noinspection PyUnresolvedReferences
from langchain.vectorstores import Pinecone, FAISS

from agents.linkedin import get_linkedin_profile_url
from agents.twitter import get_twitter_profile_username
from parsers.pydantic import PersonalIntel

# noinspection PyUnresolvedReferences
from third_parties.linkedin import get_linkedin_profile, get_saved_linkedin_profile
from langchain.docstore import InMemoryDocstore


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

    linkedin_profile = get_saved_linkedin_profile(f"storage/linkedin/ranjan252812.json")
    # tweets = scrape_user_tweets(username=run_chain_for_twitter_username(name), num_tweets=5)
    out_parser = PydanticOutputParser(pydantic_object=PersonalIntel)
    custom_template = """
        Given the LinkedIn information {linkedin_profile} about a person, create
        1. A short summary
        2. Two interesting facts about the person
        3. A topic that may interest them
        4. Two creative Ice-breakers to open a conversation with them
        5. Two questions to ask them in job interview
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


def run_retrival_qna_chain_for_text_document(q: str) -> str:
    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"],
        environment=os.environ["PINECONE_ENVIRONMENT"],
    )
    doc_loader = TextLoader(
        "storage/text_documents/rt-2-new-model-translates-vision-and-language-into-action.txt"
    )
    document = doc_loader.load()

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=350)
    contexts = splitter.split_documents(document)
    print(f"Number of contexts created by splitter: # {len(contexts)}")

    embeddings = OpenAIEmbeddings()
    vector_store_client = pinecone.Index("rt-2-robotics")
    vector_store = Pinecone(
        vector_store_client, embedding_function=embeddings.embed_query, text_key="text"
    )
    # vector_store.add_documents(contexts)

    chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"),
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
    )
    return chain.run({"query": q})


def run_retrival_qna_chain_for_pdf_document() -> None:
    doc_loader = PyPDFLoader(
        "storage/pdf_documents/WEF_Top_10_Emerging_Technologies_of_2023.pdf"
    )
    documents = doc_loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=350, separator="\n")
    contexts = splitter.split_documents(documents)
    print(
        f"Number of contexts created by splitter: # {len(contexts)}"
    )
    embeddings = OpenAIEmbeddings()

    # Initialize the vectorstore as empty
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vector_store = FAISS(
        embedding_function=embeddings.embed_query,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id=dict(),
    )

    vector_store.add_documents(contexts)

    chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"),
        chain_type="stuff",
        retriever=vector_store.as_retriever(
            search_type="mmr", search_kwargs={"k": 10, "fetch_k": 50}
        ),
    )
    print("Ask me anything on 'World Economic Forum's Report on Top 10 Emerging Technologies of 2023'", end="\n\n")
    while True:
        q = input()
        if q == "exit":
            break
        elif q == "":
            continue
        else:
            print(chain.run({"query": q}), end="\n\n")


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    print("Hello LangChain!")
    run_retrival_qna_chain_for_pdf_document()
