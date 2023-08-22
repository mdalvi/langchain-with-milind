import streamlit as st
from dotenv import load_dotenv as dotenv_load
from st_pages import Page, show_pages, add_page_title


def main():
    dotenv_load()

    add_page_title(
        page_title="Applied-Gen-AI - AIE-Mumbai",
        page_icon="storage/branding/page_icon.png",
    )
    st.write("")
    show_pages(
        [
            Page("streamlit_app/home.py", "Home", "🏠"),
            Page(
                "streamlit_app/demo_linkedin.py",
                "1. Generative-AI on LinkedIn Profile",
                ":large_green_circle:",
            ),
            Page(
                "streamlit_app/demo_guardrails.py",
                "2. Generative-AI on PDF",
                ":large_green_circle:",
            ),
            Page(
                "streamlit_app/demo_video.py",
                "3. Applying Generative-AI on Video",
                ":large_green_circle:",
            ),
        ]
    )


if __name__ == "__main__":
    main()
