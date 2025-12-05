# 🌟 LangGraph + Gemini RAG Chatbot (Streamlit Edition)

### *AI-Powered Document-Aware Conversational Assistant*

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

* **Google Gemini 2.5 Flash**
* **LangChain** for embeddings and toolchains
* **LangGraph** for orchestrated conversational flows
* **ChromaDB** for persistent vector search
* **Streamlit** for an interactive chat UI

The chatbot retrieves relevant document chunks, injects them into prompts, and produces **grounded, contextual, and accurate responses**.

The assistant identifies itself as **Bob**.

---

# ⚙️ Features

### 🔍 Smart Document Retrieval

Uses **ChromaDB** + **Gemini Embeddings** to find the most relevant PDF chunks.

### 🧠 Context-Aware Responses

The retrieved chunk text is inserted directly into the prompt.

### 🔄 LangGraph Orchestrated Workflow

Encapsulated node-based RAG pipeline:

* Query → Embed → Vector Search → Prompt → Gemini Response → Stream

### ⚡ Persistent Local Vector Store

Chroma provides high-speed similarity search with local file persistence.

### 🎨 Clean Streamlit UI

Features include:

* Real-time chat streaming
* File upload
* Conversation memory
* Interactive display

---

# 🧠 How It Works

The chatbot follows this pipeline:

1. User submits a question.
2. `retrieve_documents` performs vector search using ChromaDB.
3. Top 3 most relevant documents are returned.
4. The documents are injected into a prompt template.
5. Gemini generates a grounded answer.
6. LangGraph updates the conversation state.

This ensures **evidence-based answers with minimal hallucinations**.

---

# 🧩 Code Breakdown

## 1. Model Initialization

```python\
chat = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)
```

## Embeddings

```python
embeddings = GoogleGenerativeAIEmbeddings(
    model="text-embedding-004"
)
```

---

## 2. Vector Store Loader

```python
def load_vector_store(embeddings):
    return Chroma(
        embedding_function=embeddings,
        collection_name="pdf_docs",
        persist_directory="chroma_db"
    )
```

---

## 3. LangGraph State

```python
class State(MessagesState):
    pass
    documents: list[str]
```

Contains:

* `messages`
* `documents`

---

## 4. Document Retrieval Node

```python
def retrrieve_documents(state: State) -> State:
    vector_store = load_vector_store(embeddings)
    question = state["messages"][-1].content
    docs = vector_store.similarity_search(question, k=3)
    document_pages = [doc.page_content for doc in docs]
    return {"documents": document_pages}
```

---

## 5. Chat Model Node

```python
prompt_template = PromptTemplate(
    template="""
        Use the following documents to answer the question.

        Documents:
        {documents}

        Question:
        {question}

        Provide a detailed and accurate answer based on the documents.
        If the answer is not found in the documents, respond with \"I don't know.\"
    """
)
```

---

## 6. Graph Construction

```python
builder = StateGraph(State)
builder.add_node("chat_model", chat_model)
builder.add_node("retrieve_documents", retrrieve_documents)

builder.add_edge(START, "retrieve_documents")
builder.add_edge("retrieve_documents", "chat_model")
builder.add_edge("chat_model", END)
```

Workflow:

```
START → retrieve_documents → chat_model → END
```

---

## 7. Memory Mode (Optional)

```python
memory = MemorySaver()
react_graph_memory = builder.compile(checkpointer=memory)
```

---

## 8. CLI Chat Loop

```python
while True:
    user_input = input("You: ")
```

---

# 📥 Installation

## 1. Clone

```bash
git clone https://github.com/Jasman123/LangChain_Project
cd LangChain_Project
```

## 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux / macOS
venv\Scripts\activate     # Windows
```

## 3. Install Requirements

```bash
pip install -r requirements.txt
```

## 4. Configure Environment

Create `.env` file:

```env
GOOGLE_API_KEY=your_api_key_here
```

---
## 5. Build ChromaDB

Before running the chatbot or Streamlit app, you **must generate the ChromaDB vector store**.

Run the embedding script:

```bash
python load_vector.py
```

This processes your PDF files and stores embeddings inside the `chroma_db/` folder.

# ▶️ Run CLI Chatbot

```bash
python app.py
```

---

# 🌐 Run Streamlit App

```bash
streamlit run app.py
```

Features:

* Chat interface
* PDF document upload
* RAG search
* Memory

---

# 🧬 LangGraph Architecture

```
flowchart TD
    A[START] --> B[Retrieve Documents]
    B --> C[Chat Model]
    C --> D[END]
```
<img width="187" height="333" alt="graph" src="https://github.com/user-attachments/assets/0d06a0a3-286e-450f-80bb-e2e0b4660429" />


---

# 🗂 Directory Structure

```
📦 LangChain_Project
├── app.py
├── main.py
├── chroma_db/
├── data/
├── README.md
├── requirements.txt
└── .env
```

---

# 📄 Prompt Strategy

The assistant follows strict RAG rules:

**If the answer is not found in the documents, it responds:**

> "I don't know."

---

# 🧩 Example Chat

**User:**

> What does the document say about compliance testing?

**Assistant:**

> Based on the retrieved documents, the compliance testing steps include...

---

# 🛠 Tech Stack

| Layer         | Technology         |
| ------------- | ------------------ |
| LLM           | Gemini 2.5 Flash   |
| Embeddings    | text-embedding-004 |
| Vector Store  | ChromaDB           |
| Orchestration | LangGraph          |
| Prompting     | LangChain          |
| UI            | Streamlit          |

---

<img width="1723" height="639" alt="image" src="https://github.com/user-attachments/assets/4c982060-2a3f-474e-a707-5adc58809e43" />


# 🚀 Roadmap

* Add streaming in CLI
* Add citations
* Add PDF ingestion UI
* Add Ragas evaluation
* Deploy (Cloud Run / HuggingFace)
