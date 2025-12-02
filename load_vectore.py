from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, RemoveMessage
from typing import Literal
from IPython.display import Image, display
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.document_loaders import WebBaseLoader, AsyncChromiumLoader, PyPDFLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import asyncio
import os
import copy
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"


# import requests


load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
chat = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="text-embedding-004",
    timeout=30,
    max_retries=2,
)

def load_pdf(file_path: str) -> str:
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    for i in range(len(docs)):
        docs[i].page_content =' '.join(docs[i].page_content.split())

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    pages_splitter = text_splitter.split_documents(docs)
    return pages_splitter

def build_vector_store(documents):
    vector_store = Chroma.from_documents(
        documents,
        embedding=embeddings,
        collection_name="pdf_docs",
        persist_directory="chroma_db"
    )
    
    return vector_store

def load_vector_store(embeddings):
    vector_store = Chroma(
        embedding_function=embeddings,
        collection_name="pdf_docs",
        persist_directory="chroma_db"
    )
    return vector_store

document = load_pdf(r".\data_pdf.pdf")
vector_store = build_vector_store(document)
print("Vector store loaded.")
print("len of collection:", vector_store._collection.count())
resutl = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k":3, 'lambda_mult': 0.7}
)
question = 'What is main topic of this document?'
retrieved_docs = resutl.invoke(question)
print("Retrieved Documents:")
for doc in retrieved_docs:
    print("---- Document ----")
    print("page number:", doc.metadata.get('page'))
    print(doc.page_content)










