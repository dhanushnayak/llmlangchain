from apikey import key,ser_key
import chainlit as cl
import openai
import os
from langchain import PromptTemplate,OpenAI,LLMChain,LLMMathChain,SerpAPIWrapper
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent,Tool
os.environ['OPENAI_API_KEY']= key
os.environ['SERPAPI_API_KEY']=ser_key

@cl.langchain_factory(use_async=False)
def load():
    llm =  ChatOpenAI(temperature=0,streaming=True)
    llm1 = OpenAI(temperature=0,streaming=True)
    search = SerpAPIWrapper()
    llm_match_chain =  LLMMathChain.from_llm(llm=llm,verbose=True)

    tools = [
        Tool(
        name='Search',
        func=search.run,
        description="Used for Search Key"
        ),
        Tool(
        name='Calculator',
        func=llm_match_chain.run,
        description='Used for math'
        )
    ]

    return initialize_agent(
        tools=tools,llm=llm1,agent='chat-zero-shot-react-description',verbose=True
    )