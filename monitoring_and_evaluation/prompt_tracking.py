"""Prompt tracking with PromptWatch.io."""
from langchain import LLMChain, OpenAI, PromptTemplate
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

prompt_template = PromptTemplate.from_template("Finish this sentence {input}")
my_chain = LLMChain(llm=OpenAI(), prompt=prompt_template)

with PromptWatch() as pw:
    my_chain("The quick brown fox jumped over")

if __name__ == "__main__":
    pass
