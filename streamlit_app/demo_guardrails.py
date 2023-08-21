import os

import streamlit as st

from streamlit_helper import run_retrival_qna_chain_for_pdf_document
from common.functions import get_timestamped_filename

# Page re-run assignments
st.image("storage/branding/brand_logo.png", use_column_width=True)
st.write("## Demo of Guardrails in Generative-AI")
st.info(
    "Expectations: Gen-AI should refuse to answer queries beyond provided PDF context"
)

is_submission = True if "submitted" in st.session_state else False
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
# check if chain exists in the session
chain = st.session_state.chain if is_submission else None

# check new file upload
is_submission = (
    False
    if is_submission and (uploaded_file.name != st.session_state.get("file_name", ""))
    else is_submission
)

# Run-once
if not is_submission:
    if uploaded_file is not None:
        file_path = (
            f"storage/uploads/{get_timestamped_filename(os.environ['TIMEZONE'])}.pdf"
        )
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        st.session_state.file_name = uploaded_file.name
        st.success("File uploaded successfully!")

        with st.spinner(
            "Learning PDF (This may take sometime depending on the size of your document)..."
        ):
            chain = run_retrival_qna_chain_for_pdf_document(file_path)
            st.session_state.chain = chain
            # false submission to rerun the file
            st.session_state.submitted = True
        st.experimental_rerun()
else:
    # Input text field and submit button in the same form layout
    with st.form("input_form"):
        input_text = st.text_input(
            "Query PDF Document:",
            "",
            label_visibility="visible",
            placeholder="Your question here...",
        )
        submit_button = st.form_submit_button("Submit")
        if submit_button:
            if input_text:
                with st.spinner("Finding answer for your query..."):
                    result = chain.run({"query": input_text})
                    st.session_state.submitted = True
                    st.success(result)
            else:
                st.warning("Invalid query!")
