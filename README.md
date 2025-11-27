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


---

# 📘 Overview

This project implements a modern **Retrieval-Augmented Generation (RAG) chatbot** built with:

- **Google Gemini 2.5 Flash**  
- **LangChain** for embeddings and tools  
- **LangGraph** for orchestrated multi-step conversational flow  
- **ChromaDB** for persistent vector search  
- **Streamlit** for an elegant chat interface  

The bot retrieves relevant documents, injects them into the prompt, and generates accurate, contextual responses.  
The assistant identifies itself as **Bob**.

---

# 🧠 High-Level Architecture

```mermaid
flowchart TD
    A[User Query] --> B[Streamlit UI]
    B --> C[LangGraph State Machine]

    C --> D[Retrieve Documents Node]
    D -->|Top-k Relevant Chunks| E[Chat Node]

    E -->|Final Answer| B

graph LR
  A[Google Gemini 2.5 Flash] <-- prompts/messages --> E[LangGraph Chat Node]
  E --> D[LangGraph Retrieve Node]
  D --> C[Chroma DB]
  C <-- embeddings --> B[Google Embedding Model (text-embedding-004)]

  F[Streamlit Frontend] --> E
  E --> F



# 🔥 Features
🔎 Retrieval-Augmented Generation

Embeddings via text-embedding-004

ChromaDB persistent collection (chroma_db/)

MMR search for diverse & relevant results

# 🧠 LangGraph Pipeline

Two nodes:

retrieve_documents

chat_model

💬 Streamlit Chat UI

Saves history

Clean visual layout

Filters internal prompt injections

💾 Memory-Backed Execution

Powered by LangGraph’s MemorySaver.


# 📁 File Structure

project/
│-- main.py
│-- chroma_db/
│-- .env
│-- requirements.txt
│-- README.md


#🔧 Installation

1. Install dependencies
pip install -r requirements.txt

2. Setup environment variables

Create .env:

GOOGLE_API_KEY=your_api_key_here


#🧩 Code Overview
1. Chat Model (Gemini)
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

5. Chat Node
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
builder.add_node("chat_model", chat_model)
builder.add_node("retrieve_documents", retrieve_documents)

builder.add_edge(START, "retrieve_documents")
builder.add_edge("retrieve_documents", "chat_model")
builder.add_edge("chat_model", END)

graph = builder.compile()

7. Streamlit UI
Setup
st.set_page_config(page_title="AI RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 RAG-Powered AI Chatbot")

Rendering Chat
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage) and not msg.content.startswith("Use the following documents"):
        st.chat_message("user").write(msg.content)
    elif msg.type == "ai":
        st.chat_message("assistant").write(msg.content)
