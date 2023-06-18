import os
from apikey import key
from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain


os.environ["OPENAI_API_KEY"]  = key


demo_template = """
I want you to act as a Assistance Lawyer for people in an easy way, for the scenario {case} along with section number to file a case in india.
"""

prompt =  PromptTemplate(
    input_variables=['case'],
    template=demo_template
)





llm =  OpenAI(temperature=0.7)

chain = LLMChain(llm=llm,prompt=prompt)
res = chain.run("One of the person hitted my child")

print(res)