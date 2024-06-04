"""Tracing of agent calls and intermediate results."""
import subprocess

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import StructuredTool
from langchain.agents import initialize_agent, AgentType, create_openai_functions_agent, AgentExecutor

from langchain.pydantic_v1 import BaseModel, Field
from urllib.parse import urlparse

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

set_environment()

class PingInput(BaseModel):
    url: str = Field(description="url to ping")
    return_error: bool = Field(description="return error if ping fails")

def ping(url: str, return_error: bool = False) -> str:
    """Ping the fully specified url. Must include https:// in the url."""
    hostname = urlparse(url).netloc
    completed_process = subprocess.run(
        ["ping", "-c", "1", hostname], capture_output=True, text=True
    )
    output = completed_process.stdout
    if return_error and completed_process.returncode != 0:
        return completed_process.stderr
    return output


ping_tool = StructuredTool.from_function(
    func=ping, 
    name="ping",
    args_schema=PingInput,
    )


llm = ChatOpenAI(model="gpt-3.5-turbo-0613", temperature=0)
tools = [ping_tool]
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Answer all questions to the best of your ability."),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
agent = create_openai_functions_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
result = agent_executor.invoke({"input": "What's the latency like for https://langchain.com ?",
                                "agent_scratchpad": "Use the ping tool to ping https://langchain.com"})
print(result)


if __name__ == "__main__":
    pass
