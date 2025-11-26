# Project README

## Overview

This project implements a retrieval-augmented chatbot using LangChain, LangGraph, Google Generative AI, and Chroma for vector storage. It allows users to interact with a chatbot that retrieves relevant information from stored documents and produces context-aware responses.

## Features

* **Google Gemini 2.5 Flash** model for fast and accurate responses.
* **RAG (Retrieval-Augmented Generation)** enabled using Chroma vector store.
* **LangGraph state management** for structured conversational flows.
* **Embeddings using text-embedding-004** for efficient document retrieval.
* **Memory-backed graph execution** via `MemorySaver`.

## File Structure

```
project/
│-- main.py
│-- chroma_db/
│-- .env
│-- requirements.txt
│-- README.md
```

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Required environment variables:

```
GOOGLE_API_KEY=your_api_key_here
```

## How It Works

1. Loads Google Generative AI model for chat.
2. Loads embeddings and initializes a Chroma vector store.
3. Defines a LangGraph workflow with two main nodes:

   * **retrieve_documents** – retrieves top-k similar documents.
   * **chat_model** – constructs a prompt using retrieved documents and queries the model.
4. Runs an interactive loop allowing the user to chat with the RAG-enabled assistant.

## Running the Application

Start the chatbot with:

```bash
python main.py
```

Then type any question. To exit:

```
exit
```

## Key Code Snippets

### Loading the Vector Store

```python
def load_vector_store(embeddings):
    return Chroma(
        embedding_function=embeddings,
        collection_name="pdf_docs",
        persist_directory="chroma_db"
    )
```

### Retrieval Node

```python
def retrrieve_documents(state):
    vector_store = load_vector_store(embeddings)
    question = state["messages"][-1].content
    docs = vector_store.similarity_search(question, k=3)
    return {"documents": [doc.page_content for doc in docs]}
```

### Chat Node

```python
def chat_model(state):
    documents = "\n\n".join(state["documents"])
    question = state["messages"][-1].content
    prompt = PromptTemplate(...)
    response = chat.invoke(state['messages'])
    return {"messages": state['messages'] + [response]}
```

## Notes

* The assistant identifies itself as **Bob** per the system message.
* If no answer is found in the documents, the model responds with *"I don't know."*

## Future Improvements

* Add UI (Streamlit/Gradio).
* Include PDF ingestion pipeline.
* Improve system prompt engineering.
* Add conversation history persistence.

## License

MIT License.
