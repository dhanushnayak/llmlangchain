import os
from apikey import key
from utlis import clean_string
os.environ["OPENAI_API_KEY"] = key

from langchain.document_loaders.youtube import YoutubeLoader,GoogleApiYoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

url = 'https://youtu.be/Y9mxx1Mer0I'

loader = YoutubeLoader.from_youtube_url(youtube_url=url,add_video_info=True)

doc = loader.load()
params = {"chunk_size":2000,"chunk_overlap":0,"length_function":len}
chunk_text =  RecursiveCharacterTextSplitter(
**params
)

mytext = clean_string(doc[0].page_content)

chucked_data = chunk_text.split_text(mytext)

embedding = OpenAIEmbeddings()
docsearch = FAISS.from_texts(texts=chucked_data,embedding=embedding)

llm = OpenAI(temperature=0.7)
chain = load_qa_chain(llm=llm)


text_case = "class objects"

docs =  docsearch.similarity_search(text_case)

resp =  chain.run(input_documents = docs, question=text_case)

print(f"\n Query : {text_case}\n Resp : {resp}")

