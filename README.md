# 🧠 Local RAG System (100% Offline)

A fully local **Retrieval-Augmented Generation (RAG)** system built with:

- Ollama (local LLM)
- ChromaDB (vector database)
- LangChain (orchestration)
- HuggingFace embeddings

No OpenAI. No API keys. No internet required after setup.

---

# 🚀 What This Project Does

This project allows you to **ask questions about your own PDFs**.

Instead of guessing answers, the system:

1. Reads your PDF
2. Breaks it into smaller chunks
3. Converts text into numerical vectors (embeddings)
4. Stores them in a local database
5. Retrieves the most relevant chunks when you ask a question
6. Uses a local LLM to generate an answer grounded in your data

---

# 🧩 How RAG Works (Simple Explanation)

Think of it like an **open-book exam**:

- The LLM = student  
- Your PDF = textbook  
- Vector DB = index to quickly find relevant pages  

When you ask a question:
1. Your question is converted into a vector
2. The system finds the most similar chunks in your PDF
3. Those chunks are passed to the LLM
4. The LLM generates an answer based on that context

---

# 📁 Project Structure
