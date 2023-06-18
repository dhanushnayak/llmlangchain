
import os
from apikey import key
import chainlit as cl
import openai
from langchain import PromptTemplate,OpenAI,LLMChain
os.environ['OPENAI_API_KEY']= key

template =  """Question: {question}
Answer: Let's think step by step.
"""

@cl.langchain_factory(use_async=True)
def factory():
    promt = PromptTemplate(template=template,input_variables=['question'])
    llm_chain =  LLMChain(prompt=promt,llm=OpenAI(temperature=0.5),verbose=True)
    return llm_chain