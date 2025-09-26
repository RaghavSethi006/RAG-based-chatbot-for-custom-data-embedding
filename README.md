📚 Streamlit RAG Chatbot App with Groq

This project is a Retrieval-Augmented Generation (RAG) pipeline built with Streamlit, LangChain, FAISS, and Groq LLMs.
It allows you to upload PDF documents, embed and store them, and query the content using a powerful language model backend.

🚀 Features

Upload and parse PDF documents with PyMuPDF (fitz)

Split text using RecursiveCharacterTextSplitter

Generate vector embeddings with HuggingFaceEmbeddings

Store and query embeddings in a FAISS vector database

Ask natural language questions and get context-aware answers via Groq LLMs

Simple, interactive Streamlit web UI

🛠️ Tech Stack

Streamlit
 – UI framework

LangChain
 – RAG pipeline utilities

FAISS
 – Vector search

HuggingFace Embeddings
 – Embeddings

Groq
 – LLM inference

PyMuPDF
 – PDF parsing

📂 Project Structure
.
├── app.py              # Main Streamlit app  
├── requirements.txt    # Dependencies  
├── .env                # API keys (Groq, HuggingFace, etc.)  
└── vectorstore/        # FAISS index (auto-created after upload)  

4. Add environment variables

Create a .env file:

GROQ_API_KEY=your_groq_api_key
HF_MODEL=sentence-transformers/all-MiniLM-L6-v2

5. Run the app
streamlit run app.py

🎯 Usage

Open the Streamlit app in your browser

Upload one or more PDF files

Ask questions about the documents

Get AI-powered answers with citations from your PDFs

🔮 Future Improvements

Add support for multiple embedding models

Implement persistent vector storage (e.g., Pinecone, Weaviate)

Add document summarization and chat history
