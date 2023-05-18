import os
from apikey import key

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain,SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

os.environ["OPENAI_API_KEY"]  = key

st.title(" Custom LLM On DL")

prompt =  st.text_input("Query : ")

#Input Templates
title_template = PromptTemplate(
    input_variables= ['topic'],
    template='write me a information about {topic}'
)

example_template = PromptTemplate(
    input_variables= ['title','wikipedia_research'],
    template='write me a simple example on this title TITLE : {title}, while leveraging this wikipedia research: {wikipedia_research} '
)

#Memory
title_memory = ConversationBufferMemory(input_key= 'topic', memory_key= 'chat_history')
example_memory = ConversationBufferMemory(input_key= 'title', memory_key= 'chat_history')

#LLMs
llm = OpenAI(temperature=0.9)
title_chain =  LLMChain(llm=llm,
                        prompt=title_template,
                        verbose=1,
                        output_key='title',
                        memory=title_memory)
example_chain =  LLMChain(llm=llm,
                          prompt=example_template,
                          verbose=1,
                          output_key='example',
                          memory=example_memory)
#seq_chain =  SequentialChain(chains=[title_chain,example_chain],
                             #input_variables=['topic'],
                             #output_variables=['title','example'],verbose=1) ## Run first chain then 2nd chain ie, info of c1 as input to c2

wiki = WikipediaAPIWrapper()

if prompt:
    title =  title_chain.run(prompt) #run Chain
    wiki_res =  wiki.run(prompt)
    example  = example_chain.run(title=title,wikipedia_research=wiki_res) #run Chain
    #resp  =  seq_chain({"topic":prompt}) ## RUN Sequence
    #st.write(resp['title'])
    #st.write(resp['example'])"""

    st.write(title)
    st.write(example)

    with st.expander("Title History"):
        st.info(title_memory.buffer)

    with st.expander("Example History"):
        st.info(example_memory.buffer)
    
    with st.expander("Wiki Results"):
        st.info(wiki_res)