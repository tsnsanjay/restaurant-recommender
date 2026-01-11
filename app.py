import streamlit as st
import os
import tempfile
import gc
import base64
import time
import yaml
from tqdm import tqdm
from brightdata_scrapper import *

# Load LLM and Agents using st.secrets for safety on Cloud
from crewai import Agent, Crew, Process, Task, LLM
from crewai_tools import FileReadTool

# Ensure the transcripts directory exists locally for the app to write into
os.makedirs("transcripts", exist_ok=True)

docs_tool = FileReadTool()

# Use st.secrets instead of os.getenv for Cloud deployment
bright_data_api_key = st.secrets.get("BRIGHT_DATA_API_KEY")

@st.cache_resource
def load_llm():
    # Attempt to get OpenAI key from st.secrets
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        st.error("Missing OPENAI_API_KEY in Streamlit Secrets.")
        st.stop()
    return LLM(model="gpt-4o", api_key=api_key)

# ===========================
#   Define Agents & Tasks
# ===========================
def create_agents_and_tasks():
    """Creates a Crew for analysis of the channel scrapped output"""
    config_path = "config.yaml"
    
    # Path safety: verify file exists before attempting to open
    if not os.path.exists(config_path):
        st.error(f"Error: {config_path} not found in the repository.")
        st.stop()

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    analysis_agent = Agent(
        role=config["agents"][0]["role"],
        goal=config["agents"][0]["goal"],
        backstory=config["agents"][0]["backstory"],
        verbose=True,
        tools=[docs_tool],
        llm=load_llm()
    )

    response_synthesizer_agent = Agent(
        role=config["agents"][1]["role"],
        goal=config["agents"][1]["goal"],
        backstory=config["agents"][1]["backstory"],
        verbose=True,
        llm=load_llm()
    )

    analysis_task = Task(
        description=config["tasks"][0]["description"],
        expected_output=config["tasks"][0]["expected_output"],
        agent=analysis_agent
    )

    response_task = Task(
        description=config["tasks"][1]["description"],
        expected_output=config["tasks"][1]["expected_output"],
        agent=response_synthesizer_agent
    )

    return Crew(
        agents=[analysis_agent, response_synthesizer_agent],
        tasks=[analysis_task, response_task],
        process=Process.sequential,
        verbose=True
    )

# ===========================
#   Streamlit UI Setup
# ===========================
# Helper to safely load images
def get_base64_bin_help(bin_file):
    if os.path.exists(bin_file):
        with open(bin_file, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    return ""

crew_img = get_base64_bin_help("assets/crewai.png")
bright_img = get_base64_bin_help("assets/brightdata.png")

st.markdown(f"""
    # YouTube Trend Analysis powered by 
    <img src="data:image/png;base64,{crew_img}" width="120" style="vertical-align: -3px;"> & 
    <img src="data:image/png;base64,{bright_img}" width="120" style="vertical-align: -3px;">
""", unsafe_allow_html=True)

# ... (rest of your logic remains the same)
