import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import docx
from apikey import key
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
os.environ["OPENAI_API_KEY"]  = key
load_dotenv()

def read_pdf(path):
    with open(path,'rb') as f:
        pdf_reader = PdfReader(f)
        text=''
        for pag in range(100):
            text+=pdf_reader.pages[pag].extract_text()
    return text

def read_word(path):
    doc =  docx.Document(path)
    text=''
    for para in doc.paragraphs: text+=para.text + '\n'
    return text

def read_txt(path):
    with open(path,"r") as f:
        text = f.read()
    return text

def read_documents_from_dir(dir):
    comb_text = ''
    for file in os.listdir(dir):
        filepath = os.path.join(dir,file)
        if filepath.endswith('.pdf'): comb_text+=read_pdf(filepath)
        if filepath.endswith('.docx'): comb_text+=read_word(filepath)
        if filepath.endswith('.txt'): comb_text+=read_txt(filepath)
    return comb_text

train_directory  = 'train_files/'

text = read_documents_from_dir(train_directory)

char_text_split = CharacterTextSplitter(
    separator='\n',
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len

)

text_chunks =  char_text_split.split_text(text)


embeddings = OpenAIEmbeddings()
docsearch =  FAISS.from_texts(text_chunks,embedding=embeddings)

llm = OpenAI()
chain = load_qa_chain(llm=llm,chain_type='stuff')

query =  "use of kmeans"

docs =  docsearch.similarity_search(query)
print("DOCS \n",docs)

resp =  chain.run(input_documents = docs, question=query)

print(f"\n Query : {query}\n Resp : {resp}")
