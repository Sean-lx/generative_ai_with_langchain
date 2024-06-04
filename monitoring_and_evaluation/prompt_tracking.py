"""Prompt tracking with PromptWatch.io."""
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from promptwatch import PromptWatch

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

os.environ['PROMPTWATCH_API_KEY'] = 'N3dyYUVxRjJGeFhlU0tsS2ZES2pmNWxZUnV3MjphNzBmMmYzOC0yYjFlLTUxMTQtOTVmOC1hMjQzNDRlOGFkOTU='

prompt_template = PromptTemplate.from_template("Finish this sentence {input}")
llm = OpenAI()
runnable = prompt_template | llm

with PromptWatch() as pw:
    runnable.invoke({"input": "The quick brown rabbit jumped over"})

if __name__ == "__main__":
    pass
