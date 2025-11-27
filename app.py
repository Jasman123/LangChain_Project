from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, RemoveMessage
from typing import Literal
from IPython.display import Image, display
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
import streamlit as st

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

def load_vector_store(embeddings):
    vector_store = Chroma(
        embedding_function=embeddings,
        collection_name="pdf_docs",
        persist_directory="chroma_db"
    )
    return vector_store

class State(MessagesState):
    pass
    documents : list[str]

def retrieve_documents(state:State) -> State:
    vector_store = load_vector_store(embeddings)
    question = state["messages"][-1].content
    retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k":3, 'lambda_mult': 0.7}
    )
    docs = retriever.invoke(question)
    document_pages = [doc.page_content for doc in docs]
    return {"documents": document_pages}


def chat_model(state: State) -> State:
    documents = "\n\n".join(state["documents"])
    question = state["messages"][-1].content
    prompt_template = PromptTemplate(
        input_variables=["documents", "question"],
        template="""
        Use the following documents to answer the question.
        
        Documents:
        {documents}
        
        Question:
        {question}
        
        Provide a detailed and accurate answer based on the documents.
        If the answer is not found in the documents, respond with "I don't know."
        """
    )
    prompt = prompt_template.format(documents=documents, question=question)
    state['messages'].append(HumanMessage(content=prompt))
    # print(state['messages'])
    response = chat.invoke(state['messages'])
    return {"messages": state['messages'] + [response]}


builder = StateGraph(State)
builder.add_node("chat_model", chat_model)
builder.add_node("retrieve_documents", retrieve_documents)

builder.add_edge(START, "retrieve_documents")
builder.add_edge("retrieve_documents", "chat_model")
builder.add_edge("chat_model", END)

graph = builder.compile()
memory = MemorySaver()
react_graph_memory = builder.compile(checkpointer=memory)


if __name__ == "__main__":

    st.set_page_config(page_title="AI RAG Chatbot", page_icon="🤖", layout="wide")
    st.title("🤖 RAG-Powered AI Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = [
        SystemMessage(content="You are a helpful assistant! Your name is Bob.")
        
    ]

    user_input = st.chat_input("Ask a question about the documents...")

    if user_input:
        st.session_state.messages.append(HumanMessage(content=user_input))

    # Run the graph
        output = graph.invoke(
        {"messages": st.session_state.messages},
        config={"configurable": {"thread_id": "session_1"}}
        )

        ai_response = output["messages"][-1].content
        st.session_state.messages = output["messages"]

        # print(st.session_state.messages)

    # Display chat history
    for msg in st.session_state.messages:
        if isinstance(msg, HumanMessage) and not msg.content.strip().startswith("Use the following documents"):
            st.chat_message("user").write(msg.content)
        elif msg.type == "ai":
            st.chat_message("assistant").write(msg.content)




