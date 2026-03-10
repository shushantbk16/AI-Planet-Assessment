import os 
from pydantic import BaseModel,Field 
from typing import List 
from langchain_core.prompts import ChatPromptTemplate 
from langchain_groq import ChatGroq
from dotenv import load_dotenv


load_dotenv()


class StructuredOutput(BaseModel):
    problem_text: str = Field(description="The cleaned and fully structured text of the math problem")
    topic: str = Field(description="The main topic of the math problem, e.g. algebra, probability, calculus, linear_algebra")
    variables: List[str] = Field(description="List of math variables present in the problem, like x, y, a, b")
    constraints: List[str] = Field(description="List of constraints or equations given in the problem, e.g., 'x > 0', 'a + b = 5'")
    needs_clarification: bool = Field(description="True if the input is too ambiguous or missing crucial information to be a solvable problem, otherwise False")



def get_parser_agent():
    llm =ChatGroq(model="llama-3.3-70b-versatile",temperature=0.0  )

    structured_llm=llm.with_structured_output(StructuredOutput) 

    prompt=ChatPromptTemplate.from_template("""
    You are a Math Problem Parser Agent. Your job is to take raw text input (from OCR, ASR, or direct typing) and convert it into a structured format.

    Input Text: {input_text}

    Please analyze the text and return a JSON object with the following fields:
    1. problem_text: The cleaned and fully structured text of the math problem.
    2. topic: The main topic of the math problem (e.g., algebra, probability, calculus, linear_algebra).
    3. variables: List of math variables present in the problem (e.g., x, y, a, b).
    4. constraints: List of constraints or equations given in the problem (e.g., 'x > 0', 'a + b = 5').
    5. needs_clarification: True if the input is too ambiguous or missing crucial information to be a solvable problem, otherwise False.
    """)

    parser_chain=prompt|structured_llm
    return parser_chain


if __name__=="__main__":
    chain=get_parser_agent()
    test_text="find the limit of x square plus two x as x aproaches zero"
    res=chain.invoke({"input_text":test_text})
    print(res)