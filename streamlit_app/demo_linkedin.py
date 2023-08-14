import base64
import time

import streamlit as st
from pydantic import ValidationError, constr, BaseModel

from app_helper import run_chain_for_linkedin, GENDER_PREFIX


class LinkedInUrlModel(BaseModel):
    url: constr(regex=r"^https://www\.linkedin\.com/.*")


def get_base64_of_image(image_path):
    with open(image_path, "rb") as img_file:
        img_bytes = img_file.read()
    return base64.b64encode(img_bytes).decode()


st.image("storage/branding/brand_logo.png", use_column_width=True)
st.write("## Generative-AI on LinkedIn Profile")
st.info(
    "Expectations: Gen-AI should summarize, generate interesting facts, ice-breakers for any given Linkedin Profile."
)

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
        selected_prefix = st.selectbox("Choose Their Prefix:", GENDER_PREFIX)

if submit_button:
    if input_text:
        if selected_prefix:
            try:
                _ = LinkedInUrlModel(url=input_text)
                st.success("Valid LinkedIn URL!")
                with st.spinner("Fetching LinkedIn Profile..."):
                    obj_, dict_ = run_chain_for_linkedin(input_text, selected_prefix)
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

                        """
                    )

                with st.spinner("Generate a short summary of their profile..."):
                    time.sleep(5)
                    st.markdown("----------------------------------------------------")
                    st.write("### Short Summary")
                    st.write(f"{obj_.summary}")

                # Interesting facts
                with st.spinner("Find two interesting facts about them..."):
                    time.sleep(5)

                    st.markdown("----------------------------------------------------")
                    st.write("### Interesting Facts")
                    for fact in obj_.facts:
                        st.write(f"- {fact}")

                with st.spinner("Suggest a topic that may interest them..."):
                    time.sleep(5)

                    st.markdown("----------------------------------------------------")
                    st.write("### Topics of Interest")
                    for topic in obj_.topics_of_interest:
                        st.write(f"- {topic}")

                with st.spinner(
                    "Suggest two creative ice-breakers to open conversation with them..."
                ):
                    time.sleep(5)

                    st.markdown("----------------------------------------------------")
                    st.write("### Ice Breakers")
                    for ice in obj_.ice_breakers:
                        st.write(f"- {ice}")
                    st.markdown("----------------------------------------------------")

                    # ## Ice Breakers
                    # st.write("### Ice Breakers!")
                    # with st.expander("Spoiler!"):
                    #     for ice in obj_.ice_breakers:
                    #         st.write(f"- {ice}")
            except ValidationError as ex:
                st.error(
                    f"Invalid LinkedIn URL. Please enter a valid LinkedIn URL.{ex}"
                )
        else:
            st.warning("Select prefix for the LinkedIn Profile.")
    else:
        st.warning("URL cannot be empty. Please enter a valid LinkedIn URL.")
