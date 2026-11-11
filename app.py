"""Streamlit web interface for the hybrid YouTube AI summarizer."""

import os

import streamlit as st
from dotenv import load_dotenv

from providers import BedrockProvider, GeminiProvider, LLMError, LLMProvider, OllamaProvider
from summarizer import stream_summary
from utils import TranscriptError, extract_video_id, get_transcript

load_dotenv()  # read .env once, when the app starts

OLLAMA = "🖥️ Ollama (local & private)"
GEMINI = "☁️ Gemini (cloud & fast)"
BEDROCK = "🪨 Bedrock (cloud & AWS)"


def build_provider(choice: str) -> LLMProvider:
    """Create the provider selected in the sidebar, configured from .env."""
    if choice == OLLAMA:
        return OllamaProvider(
            model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
            host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        )
    if choice == BEDROCK:
        return BedrockProvider(
            api_key=os.getenv("AWS_BEARER_TOKEN_BEDROCK", ""),
            model=os.getenv("BEDROCK_MODEL", "anthropic.claude-sonnet-5-5"),
            region=os.getenv("AWS_REGION", "us-east-1"),
        )
    return GeminiProvider(
        api_key=os.getenv("GEMINI_API_KEY", ""),
        model=os.getenv("GEMINI_MODEL", ""),
    )


@st.cache_data(show_spinner=False)
def cached_transcript(video_id: str) -> str:
    """Download each transcript only once; errors are not cached."""
    return get_transcript(video_id)


st.set_page_config(page_title="YouTube AI Summarizer", page_icon="🎬")
st.title("🎬 YouTube AI Summarizer")
st.caption("Summarize any YouTube video locally with Ollama or in the cloud with Gemini.")

# --- Sidebar: settings ---
with st.sidebar:
    st.header("⚙️ Settings")
    engine = st.radio("AI engine", [OLLAMA, GEMINI, BEDROCK])
    language = st.selectbox("Summary language", ["English", "Romanian"])

# --- Main area: input ---
url = st.text_input("YouTube link", placeholder="https://www.youtube.com/watch?v=...")
video_id = extract_video_id(url) if url else None

if url and not video_id:
    st.error("This doesn't look like a valid YouTube link.")
if video_id:
    st.video(f"https://www.youtube.com/watch?v={video_id}")

clicked = st.button("✨ Generate summary", type="primary", disabled=not video_id)
if clicked and video_id:
    try:
        with st.spinner("Downloading transcript..."):
            transcript = cached_transcript(video_id)
        st.session_state.pop("summary", None)  # hide the previous summary
        st.divider()
        with st.spinner(f"Summarizing with {engine}..."):
            # Show the text while the model writes it; returns the full text at the end
            summary = st.write_stream(stream_summary(transcript, build_provider(engine), language))
        # session_state survives reruns: the summary stays on screen after any click
        st.session_state["summary"] = summary
        st.session_state["summary_video_id"] = video_id
        st.rerun()  # redraw the page once, from session_state, with the download button
    except (TranscriptError, LLMError) as e:
        st.error(str(e))

# --- Result: shown on every rerun while it exists ---
if "summary" in st.session_state:
    st.divider()
    st.markdown(st.session_state["summary"])
    st.download_button(
        "⬇️ Download as Markdown",
        data=st.session_state["summary"],
        file_name=f"summary_{st.session_state['summary_video_id']}.md",
        mime="text/markdown",
        on_click="ignore",  # downloading doesn't need to rerun the app
    )