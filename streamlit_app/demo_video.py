import re

import streamlit as st

from app_helper import run_retrival_qna_chain_for_video
from tools.regex import get_youtube_video_id


def update_time_offset(seconds):
    st.session_state.time_offset = seconds
    st.experimental_rerun()


st.image("storage/branding/brand_logo.png", use_column_width=True)
st.write("## Applying Generative-AI on Video")
st.info(
    "Expectations: Gen-AI should be able to answer queries based on video and also seek the video to relevant answer."
)

video_url = f"storage/videos/BZD6PBnF6F0.mp4"
video_id = get_youtube_video_id(video_url)
time_offset = int(st.session_state.get("time_offset", 0))
video_file = open(video_url, "rb")
video_bytes = video_file.read()

is_refresh = True if "video_chain" in st.session_state else False
if not is_refresh:
    with st.spinner("Loading video and doing some calculations..."):
        chain = run_retrival_qna_chain_for_video(video_id)
        st.session_state.video_chain = chain
        st.experimental_rerun()
else:
    with st.empty():
        st.video(video_bytes, start_time=time_offset)
    chain = st.session_state.video_chain
    if st.session_state.get("result", None):
        st.success(st.session_state.result)

    with st.form("input_form"):
        st.write("")
        input_text = st.text_input(
            "Query Video:",
            "",
            label_visibility="visible",
            placeholder="Your question here...",
        )
        submit_button = st.form_submit_button("Submit", use_container_width=True)

    if submit_button:
        if input_text:
            with st.spinner("Finding answer for your query..."):
                # pre-processing chain input
                input_text = re.sub(
                    r"\bgen\b", "Generative", input_text, flags=re.IGNORECASE
                )

                result = chain({"query": input_text}, return_only_outputs=True)
                source_documents = result["source_documents"]
                st.session_state.time_offset = source_documents[0].metadata["start"]
                st.session_state.result = result["result"]
                st.experimental_rerun()
        else:
            st.warning("Invalid query!")
