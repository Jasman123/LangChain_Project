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



# 🧱 Project Structure

