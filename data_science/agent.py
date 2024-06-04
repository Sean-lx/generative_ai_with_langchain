"""Agent functionality."""
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor
from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd

import os, sys
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)

# adding the parent directory to
# the sys.path.
sys.path.append(parent)

# now we can import the module in the parent
# directory.
from config import set_environment
from data_science.prompts import PROMPT

set_environment()


def create_agent(csv_file: str) -> AgentExecutor:
    """
    Create data agent.

    Args:
        csv_file: The path to the CSV file.

    Returns:
        An agent executor.
    """
    llm = OpenAI()
    df = pd.read_csv(csv_file)
    return create_pandas_dataframe_agent(llm, df, verbose=True)


def query_agent(agent: AgentExecutor, query: str) -> str:
    """Query an agent and return the response."""
    prompt = PromptTemplate(template=PROMPT, input_variables=["query"])
    return agent.run(prompt.format(query=query))
