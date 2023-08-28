# noinspection PyUnresolvedReferences
import json
import os
from pathlib import Path

import faiss

# noinspection PyUnresolvedReferences
import joblib
import pinecone

# noinspection PyUnresolvedReferences
import whisper
from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.agents.agent_toolkits import create_python_agent, create_csv_agent
from langchain.chains import LLMChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.docstore import InMemoryDocstore
from langchain.document_loaders import TextLoader, PyPDFLoader, ReadTheDocsLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.tools import PythonREPLTool

# noinspection PyUnresolvedReferences
from langchain.vectorstores import Pinecone, FAISS

# noinspection PyUnresolvedReferences
from pytube import YouTube

from agents.linkedin import get_linkedin_profile_url
from agents.twitter import get_twitter_profile_username
from parsers.pydantic import PersonalIntel

# noinspection PyUnresolvedReferences
from third_parties.linkedin import get_linkedin_profile, get_saved_linkedin_profile
from tools.regex import get_youtube_video_id


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


def run_chain_for_social_media(url: str = None, name: str = None) -> PersonalIntel:
    """
    Step 1: Searches google for LinkedIn URL of the person
    Step 2: Fetches public data from LinkedIn about the person using the url
    Step 3: Searches google for Twitter URL of the person
    Step 4: Scrapes latest tweets about the person from Twitter API
    Step 5: Generates summary and two interesting facts about the person using LLM and above information
    :return: None
    """
    url = run_chain_for_linkedin_url(name) if url is None else url
    profile = get_linkedin_profile(profile_url=url)
    pi = profile["public_identifier"]
    with open(f"storage/linkedin/{pi}.json", "w") as f:
        f.write(json.dumps(profile))

    linkedin_profile = get_saved_linkedin_profile(f"storage/linkedin/{pi}.json")
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


def run_retrival_qna_chain_for_text_document(q: str) -> str:
    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"],
        environment=os.environ["PINECONE_ENVIRONMENT"],
    )
    doc_loader = TextLoader(
        "storage/text_documents/rt-2-new-model-translates-vision-and-language-into-action.txt"
    )
    raw_documents = doc_loader.load()

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=350)
    contexts = splitter.split_documents(raw_documents)
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
    """
    Demonstrates guard rails using prompt engineering
    :return:
    """
    doc_loader = PyPDFLoader("storage/pdf_documents/Capgemini-Gen-AI-portfolio_ENG.pdf")
    raw_documents = doc_loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=350, separator="\n")
    contexts = splitter.split_documents(raw_documents)
    print(f"Number of contexts created by splitter: # {len(contexts)}")
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

    prompt_template = """Use the following pieces of context to answer the question at the end. If the answer is not available in the provided context, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Answer:"""
    custom_template = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": custom_template}
    chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"),
        chain_type="stuff",
        retriever=vector_store.as_retriever(
            search_type="mmr", search_kwargs={"k": 10, "fetch_k": 50}
        ),
        chain_type_kwargs=chain_type_kwargs,
    )
    print(
        "Ask me anything on the provided PDF",
        end="\n\n",
    )
    while True:
        q = input()
        if q == "exit":
            break
        elif q == "":
            continue
        else:
            print(chain.run({"query": q}), end="\n\n")


def run_chain_on_read_the_docs() -> None:
    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"],
        environment=os.environ["PINECONE_ENVIRONMENT"],
    )
    doc_loader = ReadTheDocsLoader(
        path="storage/documentations/deap/deap.readthedocs.io/",
        features="html.parser",
    )

    raw_documents = doc_loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    contexts = splitter.split_documents(raw_documents)
    print(f"Number of contexts created by splitter: # {len(contexts)}")
    embeddings = OpenAIEmbeddings()
    vector_store_client = pinecone.Index("deap-docs")
    vector_store = Pinecone(
        vector_store_client, embedding_function=embeddings.embed_query, text_key="text"
    )
    vector_store.add_documents(contexts)

    chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"),
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
    )
    print("Ask me anything on 'DEAP Documentation'", end="\n\n")
    while True:
        q = input()
        if q == "exit":
            break
        elif q == "":
            continue
        else:
            print(chain.run({"query": q}), end="\n\n")


def run_chain_on_tube(url: str):
    video_id = get_youtube_video_id(url)
    # yt = YouTube(url)
    # down_ext = "mp4"
    # yt.streams.order_by(
    #     "resolution"
    # ).desc().first().download(output_path=f"storage/videos/{video_id}.{down_ext}")
    # yt.streams.get_audio_only().download(
    #     output_path=f"storage/videos/audio_{video_id}.{down_ext}"
    # )
    model = whisper.load_model("base")
    result = model.transcribe("storage/videos/BZD6PBnF6F0.mp4")
    joblib.dump(result, f"storage/videos/{video_id}_transcript.joblib")


def run_chain_for_python_agent():
    agent = create_python_agent(
        llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"),
        verbose=True,
        tool=PythonREPLTool(),
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )

    print(
        agent.run(
            "write a python code to connect a linux server 10.0.0.1 using password ^*(JHKKLL using SSH. \
            paramiko lib is already installed on system. DO NOT EXECUTE THE CODE"
        )
    )


def run_chain_for_csv_agent():
    agent = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"),
        verbose=True,
        path="storage/csv_documents/episode_info.csv",
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )

    print(agent.run("how many records in CSV file"))


def run_chain_for_agent_router():
    py_agent = create_python_agent(
        llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"),
        verbose=True,
        tool=PythonREPLTool(),
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )
    csv_agent = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"),
        verbose=True,
        path="storage/csv_documents/episode_info.csv",
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )
    agent = initialize_agent(
        tools=[
            Tool(
                name="python agent",
                func=py_agent.run,
                description="useful when you need to transform natural language \
        instruction into python code and execute it, returning the results of code execution. \
        DO NOT SEND PYTHON CODE TO THIS TOOL.",
            ),
            Tool(
                name="python agent",
                func=csv_agent.run,
                description="useful when you need to answer questions from CSV file. \
                Takes an input in natural language, performs calculations on CSV using pandas and returns the results.",
            ),
        ],
        llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"),
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
    )
    print(
        agent.run(
            "what are the total number of records in CSV. Use the output and find the square root of it using python"
        )
    )


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    print("Hello LangChain!")
    # run_chain_on_tube("https://www.youtube.com/watch?v=BZD6PBnF6F0")
    run_chain_for_agent_router()
    # run_chain_for_social_media(url="https://www.linkedin.com/in/aiman-ezzat/")
