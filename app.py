# noinspection PyUnresolvedReferences
import json

# noinspection PyUnresolvedReferences
import joblib
import streamlit as st
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import ValidationError, constr, BaseModel
import os
from parsers.pydantic import PersonalIntel
from typing import Tuple
import shutil
# noinspection PyUnresolvedReferences
from third_parties.linkedin import get_linkedin_profile, get_saved_linkedin_profile

from tools.regex import get_linkedin_username
from tools.requests import get_image_from_url
from dotenv import load_dotenv
import base64


class LinkedInUrlModel(BaseModel):
    url: constr(regex=r"^https://www\.linkedin\.com/.*")


def get_base64_of_image(image_path):
    with open(image_path, "rb") as img_file:
        img_bytes = img_file.read()
    return base64.b64encode(img_bytes).decode()


def run_chain_for_linkedin(
    linkedin_url: str, prefix: str
) -> Tuple[PersonalIntel, dict]:
    """
    Step 1: Fetches public data from LinkedIn about the person using the url
    Step 2: Generates summary and two interesting facts about the person using LLM and above information
    :return: None
    """
    user_name = get_linkedin_username(profile_url=linkedin_url)
    print(user_name)
    file_path = f"storage/linkedin/{user_name}.json"
    if os.path.exists(file_path):
        print("Using saved profile to process request.")
        profile_data = get_saved_linkedin_profile(f"storage/linkedin/{user_name}.json")
    else:
        print("Invoking nubela.co for profile details.")
        profile = get_linkedin_profile(profile_url=linkedin_url)
        with open(f"storage/linkedin/{user_name}.json", "wb") as f:
            f.write(profile.content)
        profile_data = json.loads(profile.content.decode("utf-8"))

    image_path = f"storage/linkedin/images/{user_name}.jpg"
    if not os.path.exists(image_path):
        if not get_image_from_url(profile_data["profile_pic_url"], f"storage/linkedin/images/{user_name}.jpg"):
            shutil.copy(f"storage/linkedin/images/no_image.jpg", f"storage/linkedin/images/{user_name}.jpg")

    profile_data.pop("profile_pic_url", None)
    profile_data.pop("background_cover_image_url", None)
    profile_data.pop("recommendations", None)

    out_parser = PydanticOutputParser(pydantic_object=PersonalIntel)
    custom_template = """
        Given the LinkedIn information {profile_data} about a person, create
        1. A short summary
        2. Two interesting facts about the person, use gender prefix as {prefix}
        3. A topic that may interest them
        4. Two creative Ice-breakers to open a conversation with them
        5. Two questions to ask them in job interview
        \n{format_instructions}
        
        
        """

    prompt = PromptTemplate(
        input_variables=["profile_data", "prefix"],
        template=custom_template,
        partial_variables={"format_instructions": out_parser.get_format_instructions()},
    )
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    chain = LLMChain(llm=llm, prompt=prompt)
    return (
        out_parser.parse(chain.run(profile_data=profile_data, prefix=prefix)),
        {
            "image_path": image_path,
            "full_name": profile_data["full_name"],
            "headline": profile_data["headline"],
        },
    )


def main():
    load_dotenv()
    st.title("GenAI on LinkedIn")

    # Input text field and submit button in the same form layout
    with st.form("input_form"):
        # Create a layout with two columns
        col1, col2 = st.columns(2)
        with col1:
            input_text = st.text_input(
                "Enter LinkedIn URL:",
                "",
                label_visibility="visible",
                placeholder="LinkedIn Profile URL Here...",
            )
            submit_button = st.form_submit_button("Submit")
        with col2:
            prefixes = [
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
            selected_prefix = st.selectbox("Choose Their Prefix:", prefixes)

    if submit_button:
        if input_text:
            if selected_prefix:
                try:
                    _ = LinkedInUrlModel(url=input_text)
                    st.success("Valid LinkedIn URL!")
                    with st.spinner(
                        "Applying **GEN-AI** for some interesting facts..."
                    ):
                        obj_, dict_ = run_chain_for_linkedin(
                            input_text, selected_prefix
                        )
                        # Apply circular style using HTML and CSS
                        circular_image_html = f"""
                        <style>
                            .circle-image {{
                                border-radius: 50%;
                                width: 150px;
                                height: 150px;
                                object-fit: cover;
                            }}
                        </style>
                        <div style="display: flex; justify-content: center;">
                            <img class="circle-image" src="data:image/png;base64,{get_base64_of_image(dict_['image_path'])}">
                        </div>
                        """
                        st.markdown(circular_image_html, unsafe_allow_html=True)

                        # Summary
                        st.markdown(
                            f"""
                            **{dict_['full_name']}**
    
                            {dict_['headline']}
    
                            {obj_.summary}
                            """
                        )

                        # Interesting facts
                        st.write("### Interesting Facts")
                        for fact in obj_.facts:
                            st.write(f"- {fact}")

                        # Topics of Interest
                        st.write("### Topics of Interest")
                        for topic in obj_.topics_of_interest:
                            st.write(f"- {topic}")

                        # Ice-breakers
                        st.write("### Ice Breakers")
                        with st.expander("Spoiler!"):
                            for ice in obj_.ice_breakers:
                                st.write(f"- {ice}")

                        # Job-Interview Questions
                        st.write("### What you can ask them in Job Interview?")
                        with st.expander("Spoiler!"):
                            for question in obj_.interview_questions:
                                st.write(f"- {question}")
                except ValidationError as ex:
                    st.error(
                        f"Invalid LinkedIn URL. Please enter a valid LinkedIn URL.{ex}"
                    )
            else:
                st.warning("Select prefix for the LinkedIn Profile.")
        else:
            st.warning("URL cannot be empty. Please enter a valid LinkedIn URL.")


if __name__ == "__main__":
    main()
