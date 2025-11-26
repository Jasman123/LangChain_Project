from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, RemoveMessage
from langchain_core.prompts import PromptTemplate
from typing import Literal
from IPython.display import Image, display
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.document_loaders import WebBaseLoader, AsyncChromiumLoader, PyPDFLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
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

class State(MessagesState):
    summary: str
    context_docs: list[str]
    
# def filter_messages(state: MessagesState):
#     delele_messages = [RemoveMessage(id=msg.id) for msg in state['messages'][:-2]]

def call_model(state: State):

    summary = state.get("summary", "")

    if summary:
        system_message = f"You are a helpful assistant! Here is the conversation summary so far: {summary}"
        
        messages = [SystemMessage(content=system_message)] + state["messages"]

    else:
        messages = state["messages"]


    response = chat.invoke(messages)
    return {"messages": state["messages"] + [response]}

def summarize_conversation(state: State):
    print("\n=== SUMMARIZING CONVERSATION ===\n")
    summary = state.get("summary", "")

    if summary:
        
        # A summary already exists
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
        
    else:
        summary_message = "Create a summary of the conversation above:"

    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = chat.invoke(messages)

    # Delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    new_messages = state["messages"][-2:]
    return {"summary": response.content, "messages": delete_messages + new_messages}

def should_summarize(state: State) -> Literal["summarize",END]:
    """Return the next node to execute."""

    messages = state["messages"]

    if len(messages) > 6:
        return "summarize"
    else:
        return END

async def load_website(urls: list[str]) -> list[str]:
    loader = AsyncChromiumLoader(urls, user_agent="MyAppUserAgent")
    docs = await loader.aload()
    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(docs)
    return docs_transformed[0].page_content

def load_pdf(file_path: str) -> str:
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    for i in range(len(docs)):
        docs[i].page_content =' '.join(docs[i].page_content.split())

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    pages_splitter = text_splitter.split_documents(docs)
    # full_text = "\n".join([t.page_content for t in texts])
    return pages_splitter

def build_vector_store(documents, embeddings):
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

def retrieve(state: State): 
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":3})
    question = state.messages[-1].content
    docs = retriever.invoke(question)
    context = [doc.page_content for doc in docs]
    return {
        "context_docs": context,
        "messages": state.messages  
    }

def chat_model(state: State):
    context = "\n\n".join(state.context_docs)
    question = state.messages[-1].content

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer:
If the answer is not found in context, say:
"Maaf, saya tidak menemukan jawabannya di dokumen."
"""
    )

    prompt = prompt_template.format(context=context, question=question)

    response = chat.invoke([HumanMessage(content=prompt)])

    return {
        "messages": state["messages"] + [response],
        "context_docs": []
    }

vector_store = load_vector_store(embeddings)

builder = StateGraph(State)
builder.add_node("initial_model", initial_model)


builder.add_edge(START, "initial_model")
builder.add_edge("initial_model", END)

graph = builder.compile()
memory = MemorySaver()
react_graph_memory = builder.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "1"}}


if __name__ == "__main__":
    # documents = load_pdf("langchain_project/docs/langchain_overview.pdf")
    # vector_store = build_vector_store(documents, embeddings)

    print("RAG Chatbot ready...\n")

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        initial_state = State(
    messages=[HumanMessage(content=user_input)],
    context_docs=[]
    )
        
        state = graph.invoke(initial_state)
        response_message = state.messages[-1]
        print(f"Bot: {response_message.content}\n")












