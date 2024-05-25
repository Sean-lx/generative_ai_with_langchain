from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
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

model = ChatOpenAI()
prompt = PromptTemplate(input_variables=[
                        "text"], template="Classify the sentiment of this text: {text}")
chain = prompt | model
print(chain.invoke({"text": "I hated that movie, it was terrible!"}))


if __name__ == "__main__":
    pass
