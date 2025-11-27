🌟 LangGraph + Gemini RAG Chatbot + Streamlit
<p align="center"> <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" /> <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit" /> <img src="https://img.shields.io/badge/Google%20Gemini-2.5%20Flash-4285F4?logo=google" /> <img src="https://img.shields.io/badge/LangChain-Framework-orange?logo=chainlink" /> <img src="https://img.shields.io/badge/LangGraph-Orchestration-green" /> </p> <p align="center"><b>Retrieval-Augmented Chatbot powered by LangGraph, Google Gemini, and Chroma — deployed with a clean Streamlit UI.</b></p>
🚀 Overview

This project implements a RAG (Retrieval-Augmented Generation) chatbot using:

Google Gemini 2.5 Flash for fast, high-quality reasoning

LangChain for modular components

LangGraph for structured conversational workflows

ChromaDB for vector search

Streamlit for an interactive web UI

Users can ask questions, and the chatbot retrieves the most relevant documents from the vector store, adds them to the prompt, and generates a context-aware response.

The assistant identifies itself as Bob.

✨ Features
🔍 Retrieval Augmented Generation (RAG)

Uses GoogleGenerativeAIEmbeddings (text-embedding-004).

Chroma Vector DB with persistent storage.

MMR-based retriever for diverse top-k results.

🧠 LangGraph Conversation Flow

Two main nodes:

retrieve_documents – fetches top-k relevant chunks

chat_model – augments prompt + queries Gemini

💬 Streamlit Chat UI

Smooth chat interface with message history

Session-based memory

Auto-renders responses

📁 File Structure
project/
│-- main.py
│-- chroma_db/
│-- .env
│-- requirements.txt
│-- README.md

🔧 Requirements

Install dependencies:

pip install -r requirements.txt


Environment variables:

GOOGLE_API_KEY=your_api_key_here

🧩 Core Components
1. Google Gemini Chat Model
chat = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

2. Embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="text-embedding-004",
    timeout=30,
    max_retries=2,
)

3. Vector Store Loader
def load_vector_store(embeddings):
    return Chroma(
        embedding_function=embeddings,
        collection_name="pdf_docs",
        persist_directory="chroma_db"
    )

4. Retrieval Node (MMR Search)
def retrieve_documents(state: State) -> State:
    vector_store = load_vector_store(embeddings)
    question = state["messages"][-1].content
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "lambda_mult": 0.7}
    )
    docs = retriever.invoke(question)
    document_pages = [doc.page_content for doc in docs]
    return {"documents": document_pages}

5. Chat Node (Prompt + Generation)
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
    response = chat.invoke(state['messages'])
    return {"messages": state['messages'] + [response]}

6. LangGraph Workflow
builder = StateGraph(State)
builder.add_node("retrieve_documents", retrieve_documents)
builder.add_node("chat_model", chat_model)

builder.add_edge(START, "retrieve_documents")
builder.add_edge("retrieve_documents", "chat_model")
builder.add_edge("chat_model", END)

graph = builder.compile()

7. Streamlit Application

The UI includes:

Chat input

Persistent session state

Filter to avoid showing internal prompt injections

st.set_page_config(page_title="AI RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 RAG-Powered AI Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content="You are a helpful assistant! Your name is Bob.")]


Renders chat history:

for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage) and not msg.content.startswith("Use the following documents"):
        st.chat_message("user").write(msg.content)
    elif msg.type == "ai":
        st.chat_message("assistant").write(msg.content)

▶️ Running the Application

Start the Streamlit app with:

streamlit run main.py


Then type into the chat input.

To exit: close the Streamlit UI.

🔮 Future Improvements

Add file ingestion (PDFs, text, web URLs)

Enhanced streaming support

User-configurable retrieval settings

Model selection dropdown

Multi-session conversation storage
