from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders.youtube import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS,Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import chainlit as cl
from utlis import clean_string
import os
from apikey import key
os.environ["OPENAI_API_KEY"]  = key

params = {"chunk_size":2000,"chunk_overlap":0,"length_function":len}
chunk_text =  RecursiveCharacterTextSplitter(
**params
)

system_template = """Use the following pieces of context to answer the users question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Example of your response should be:

```
The answer is foo
SOURCES: xyz
```

Begin!
----------------
{summaries}"""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}

author_of_video =  'No User'

@cl.langchain_rename
def rename(orig_author: str):
    global author_of_video
    rename_dict = {
        "ChatOpenAI": author_of_video
    }

    return rename_dict.get(orig_author, orig_author)

@cl.langchain_factory(use_async=True)
async def init():

    global author_of_video

    url = None
    while url is None:
        url = await cl.AskUserMessage("Please provide the Youtube Url").send()

    msg = cl.Message(f"Processing Url -> {url['content']}")
    await msg.send()

    loader = YoutubeLoader.from_youtube_url(youtube_url=url['content'],add_video_info=True)
    doc = loader.load()
    print(doc)
    author_of_video = str(doc[0].dict)

    doc = clean_string(doc[0].page_content)

    chucked_data = chunk_text.split_text(doc)

    metadatas = [{"source": f"{i}-pl"} for i in range(len(chucked_data))]

    embedding = OpenAIEmbeddings()

    doc_search = await cl.make_async(Chroma.from_texts)(
        chucked_data,embedding,metadatas=metadatas
    )

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        ChatOpenAI(temperature=0),chain_type="stuff",retriever=doc_search.as_retriever(),
    )
    
    await msg.update(content="You can query your questions !!! ")
    
    cl.user_session.set("metadatas", metadatas)
    cl.user_session.set("texts",chucked_data)

    return chain

@cl.langchain_postprocess
async def process_response(res):
    print(res)
    answer = res['answer']
    sources = res['sources'].strip()

    source_elements = []

    metadatas = cl.user_session.get("metadatas")
    all_sources = [m["source"] for m in metadatas]
    texts = cl.user_session.get("texts")

    if sources:
        found_sources = []


        for source in sources.split(","):
            source_name = source.strip().replace(".", "")
      
            try:
                index = all_sources.index(source_name)
            except ValueError:
                continue
            text = texts[index]
            found_sources.append(source_name)
            # Create the text element referenced in the message
            source_elements.append(cl.Text(content=text, name=source_name))

        if found_sources:
            answer += f"\nSources: {', '.join(found_sources)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=source_elements).send()
    



