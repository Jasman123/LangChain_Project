# 🌟 LangGraph + Gemini RAG Chatbot (Streamlit Edition)  
### _AI-Powered Document-Aware Conversational Assistant_

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" />
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit" />
  <img src="https://img.shields.io/badge/Google%20Gemini-2.5%20Flash-4285F4?logo=google" />
  <img src="https://img.shields.io/badge/LangChain-Framework-orange?logo=chainlink" />
  <img src="https://img.shields.io/badge/LangGraph-Orchestration-green" />
  <img src="https://img.shields.io/badge/RAG-System-purple" />
</p>

---

# 📘 Overview

This project implements a modern **Retrieval-Augmented Generation (RAG) chatbot** built with:

- **Google Gemini 2.5 Flash**  
- **LangChain** for embeddings and tools  
- **LangGraph** for orchestrated conversational flow  
- **ChromaDB** for persistent vector search  
- **Streamlit** for a polished chat UI  

The assistant retrieves relevant documents, injects them into the prompt, and generates accurate, contextual responses.  
The assistant identifies itself as **Bob**.

---

# ⚙️ Features

### 🔍 Smart Document Retrieval
Uses **ChromaDB** + **Gemini Embeddings** to find the most relevant content.

### 🧠 Context-Aware Responses  
Documents are inserted directly into prompt context for deeply informed answers.

### 🔄 Conversation Orchestration (LangGraph)  
A multi-node workflow:
- Query → Embed → Search → Generate → Stream Result

### ⚡ Fast Local Vector Search  
Chroma provides persistent, lightweight, high-speed embedding lookup.

### 🎨 Streamlit UI  
Includes:
- Full chat interface  
- File upload panel  
- Conversation memory  
- Real-time streaming responses

# 🧠 How It Works

Your chatbot processes queries through this sequence:

1.  User submits a question\
2.  `retrieve_documents` performs vector search using **ChromaDB**\
3.  Top 3 relevant chunks are returned\
4.  A prompt is constructed and sent to **Gemini**\
5.  Gemini responds strictly based on context\
6.  LangGraph returns updated conversation state

This ensures grounded, document‑based answers.

------------------------------------------------------------------------

## 🧩 Code Breakdown

### 1. Model Initialization

``` python
chat = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)
```

### Embeddings

``` python
embeddings = GoogleGenerativeAIEmbeddings(
    model="text-embedding-004"
)
```

------------------------------------------------------------------------

### 2. Vector Store Loader

``` python
def load_vector_store(embeddings):
    return Chroma(
        embedding_function=embeddings,
        collection_name="pdf_docs",
        persist_directory="chroma_db"
    )
```

Stores vectorized PDF chunks.

------------------------------------------------------------------------

### 3. LangGraph State

``` python
class State(MessagesState):
    pass
    documents: list[str]
```

State contains: - `messages`: conversation history\
- `documents`: retrieved chunks

------------------------------------------------------------------------

### 4. Document Retrieval Node

``` python
def retrrieve_documents(state: State) -> State:
    vector_store = load_vector_store(embeddings)
    question = state["messages"][-1].content
    docs = vector_store.similarity_search(question, k=3)
    document_pages = [doc.page_content for doc in docs]
    return {"documents": document_pages}
```

------------------------------------------------------------------------

### 5. Chat Model Node

Prompt template:

``` python
prompt_template = PromptTemplate(
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
```

Gemini produces a grounded RAG answer.

------------------------------------------------------------------------

### 6. Graph Construction

``` python
builder = StateGraph(State)
builder.add_node("chat_model", chat_model)
builder.add_node("retrieve_documents", retrrieve_documents)

builder.add_edge(START, "retrieve_documents")
builder.add_edge("retrieve_documents", "chat_model")
builder.add_edge("chat_model", END)
```

Workflow:

    START → retrieve_documents → chat_model → END

------------------------------------------------------------------------

### 7. Memory Mode (Optional)

``` python
memory = MemorySaver()
react_graph_memory = builder.compile(checkpointer=memory)
```

Enables persistent chat threads.

------------------------------------------------------------------------

### 8. CLI Chat Loop

``` python
while True:
    user_input = input("You: ")
```

------------------------------------------------------------------------

## 📥 Installation

### 1. Clone Repository

``` bash
git clone https://github.com/Jasman123/LangChain_Project

cd <your-repo>
```

### 2. Create venv

``` bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

``` bash
pip install -r requirements.txt
```

### 4. Setup API Key

``` bash
GOOGLE_API_KEY=your_api_key_here
```

------------------------------------------------------------------------

## ▶️ Run CLI Chatbot

``` bash
python main.py
```

------------------------------------------------------------------------

## 🌐 Run Streamlit App

``` bash
streamlit run app.py
```

Features: - Chat interface\
- PDF upload\
- RAG responses\
- Conversation memory

------------------------------------------------------------------------

## 🧬 LangGraph Architecture

    flowchart TD
        A[START] --> B[Retrieve Documents]
        B --> C[Chat Model]
        C --> D[END]

------------------------------------------------------------------------

## 🗂 Directory Structure

    📦 rag-chatbot
    ├── app.py
    ├── main.py
    ├── chroma_db/
    ├── data/
    ├── README.md
    ├── requirements.txt
    └── .env

------------------------------------------------------------------------

## 📄 Prompt Strategy

The assistant follows strict evidence‑based reasoning:

**If the answer is not found in the documents, it responds:\
"I don't know."**

------------------------------------------------------------------------

## 🧩 Example Chat

**User:**\
*"What does the document say about compliance testing?"*

**Assistant:**\
"Based on the retrieved documents, the compliance testing steps
include..."

------------------------------------------------------------------------

## 🛠 Tech Stack

  Layer           Technology
  --------------- --------------------
  LLM             Gemini 2.5 Flash
  Embeddings      text-embedding-004
  Vector Store    ChromaDB
  Orchestration   LangGraph
  Prompting       LangChain
  UI              Streamlit

------------------------------------------------------------------------

## 🚀 Roadmap

-   Add streaming in CLI\
-   Chunk‑level citations\
-   Add PDF ingestion in UI\
-   Evaluation pipeline (Ragas)\
-   Deploy to Cloud Run / Spaces





