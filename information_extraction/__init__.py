"""Information information_extraction from documents.

The example CV is from https://github.com/xitanggg/open-resume.
"""
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_extraction_chain_pydantic
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.pydantic_v1 import BaseModel, Field

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


class Experience(BaseModel):
    # the title doesn't seem to help at all.
    start_date: Optional[str] = Field(
        description="When the job or study started.")
    end_date: Optional[str] = Field(description="When the job or study ended.")
    description: Optional[str] = Field(
        description="What the job or study entailed.")
    country: Optional[str] = Field(
        description="The country of the institution.")


class Study(Experience):
    degree: Optional[str] = Field(
        description="The degree obtained or expected.")
    institution: Optional[str] = Field(
        description="The university, college, or educational institution visited."
    )
    country: Optional[str] = Field(
        description="The country of the institution.")
    grade: Optional[str] = Field(description="The grade achieved or expected.")


class WorkExperience(Experience):
    company: str = Field(
        description="The company name of the work experience.")
    job_title: Optional[str] = Field(description="The job title.")


class Resume(BaseModel):
    first_name: Optional[str] = Field(
        description="The first name of the person.")
    last_name: Optional[str] = Field(
        description="The last name of the person.")
    linkedin_url: Optional[str] = Field(
        description="The url of the linkedin profile of the person."
    )
    email_address: Optional[str] = Field(
        description="The email address of the person.")
    nationality: Optional[str] = Field(
        description="The nationality of the person.")
    skill: Optional[str] = Field(
        description="A skill listed or mentioned in a description.")
    study: Optional[Study] = Field(
        description="A study that the person completed or is in progress of completing."
    )
    work_experience: Optional[WorkExperience] = Field(
        description="A work experience of the person."
    )
    hobby: Optional[str] = Field(
        description="A hobby or recreational activity of the person.")


def parse_cv(pdf_file_path: str) -> str:
    """Parse a resume.
    Not totally sure about the return type: is it list[Resume]?
    """

    # Define a custom prompt to provide instructions and any additional context.
    # 1) You can add examples into the prompt template to improve extraction quality
    # 2) Introduce additional parameters to take context into account (e.g., include metadata
    # about the document from which the text was extracted.)
    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value.",
        ),
        # Please see the how-to about improving performance with
        # reference examples.
        # MessagesPlaceholder('examples'),
        ("human", "{text}"),
    ]
)
    pdf_loader = PyPDFLoader(pdf_file_path)
    docs = pdf_loader.load_and_split()
    # please note that function calling is not enabled for all models!
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    runnable = prompt | llm.with_structured_output(schema=Resume)
    
    return runnable.invoke({"text": docs})


if __name__ == "__main__":
    print(parse_cv(
        pdf_file_path = current + "/" + "openresume-resume.pdf"
    ))
