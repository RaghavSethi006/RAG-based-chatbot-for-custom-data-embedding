import streamlit as st
import time
import os
import fitz  # PyMuPDF
from datetime import datetime
import json
from dotenv import dotenv_values
from groq import Groq
import warnings
warnings.filterwarnings("ignore")

# Embedding and RAG imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Load environment variables from the .env file
env_vars = dotenv_values(".env")
GroqAPIKey = env_vars.get("GroqAPIKey") 
Username = env_vars.get("Username", "User")
AssistantName = env_vars.get("AssistantName", "Assistant")

# Initialize the Groq client with the API key
if GroqAPIKey:
    groq = Groq(api_key=GroqAPIKey)

# Configure the Streamlit page
st.set_page_config(
    page_title="Custom Embedding Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dynamic CSS (keeping your original styling)
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
        border: 1px solid var(--secondary-background-color);
    }
    .user-message {
        background-color: var(--primary-color-light, #e3f2fd);
        margin-left: 20%;
    }
    .bot-message {
        background-color: var(--secondary-background-color);
        margin-right: 20%;
    }
    .message-content {
        margin-left: 10px;
        flex: 1;
        color: var(--text-color);
    }
    .message-avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 18px;
        flex-shrink: 0;
    }
    .user-avatar {
        background-color: var(--primary-color, #2196f3);
        color: white;
    }
    .bot-avatar {
        background-color: #4caf50;
        color: white;
    }
    .timestamp {
        font-size: 0.8em;
        color: var(--text-color-light, #666);
        margin-top: 5px;
        opacity: 0.7;
    }
    .source-doc {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
        font-size: 0.9em;
        border-left: 4px solid #4caf50;
    }
    
    /* Dark theme specific adjustments */
    [data-theme="dark"] .user-message {
        background-color: rgba(33, 150, 243, 0.15);
        border-color: rgba(33, 150, 243, 0.3);
    }
    
    [data-theme="dark"] .bot-message {
        background-color: var(--secondary-background-color);
        border-color: rgba(255, 255, 255, 0.1);
    }
    
    [data-theme="dark"] .source-doc {
        background-color: rgba(255, 255, 255, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# PDF Processing Functions
def extract_text_from_pdf(path):
    """Extract text from PDF file"""
    try:
        doc = fitz.open(path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text
    except Exception as e:
        st.error(f"Error extracting text from {path}: {str(e)}")
        return ""

def load_processed_files_log():
    """Load the log of already processed files"""
    log_file = "processed_files.json"
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_processed_files_log(processed_files):
    """Save the log of processed files"""
    log_file = "processed_files.json"
    with open(log_file, 'w') as f:
        json.dump(processed_files, f, indent=2)

def get_file_hash(filepath):
    """Get a simple hash of the file for change detection"""
    import hashlib
    try:
        stat = os.stat(filepath)
        # Use file size and modification time as a simple hash
        return hashlib.md5(f"{stat.st_size}_{stat.st_mtime}".encode()).hexdigest()
    except:
        return None

def process_pdfs_and_create_embeddings(pdf_folder="pdfs", force_rebuild=False):
    """Process PDFs and create/update vector embeddings intelligently"""
    if not os.path.exists(pdf_folder):
        st.warning(f"PDF folder '{pdf_folder}' not found. Please create it and add your PDF files.")
        return None
    
    # Load embedding model
    with st.spinner("Loading embedding model..."):
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    
    # Setup chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    # Load existing processed files log
    processed_files = load_processed_files_log()
    
    # Get current PDF files
    current_pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    
    if not current_pdf_files:
        st.warning(f"No PDF files found in '{pdf_folder}' folder.")
        return None
    
    # Check if we need to process any files
    files_to_process = []
    existing_vectorstore = None
    
    if not force_rebuild and os.path.exists("faiss_index"):
        # Load existing vectorstore
        try:
            existing_vectorstore = FAISS.load_local(
                "faiss_index",
                embeddings=embedding_model,
                allow_dangerous_deserialization=True
            )
            st.info("✅ Loaded existing embeddings")
        except Exception as e:
            st.warning(f"Could not load existing embeddings: {e}")
            existing_vectorstore = None
    
    # Determine which files need processing
    for filename in current_pdf_files:
        filepath = os.path.join(pdf_folder, filename)
        current_hash = get_file_hash(filepath)
        
        if (force_rebuild or 
            filename not in processed_files or 
            processed_files[filename] != current_hash):
            files_to_process.append((filename, filepath, current_hash))
    
    # Remove files that no longer exist from processed log
    processed_files = {k: v for k, v in processed_files.items() 
                      if k in current_pdf_files}
    
    # If no new files to process and existing vectorstore exists
    if not files_to_process and existing_vectorstore:
        st.success("✅ All PDF files are already processed and up-to-date!")
        return existing_vectorstore
    
    # Process new/changed files
    new_chunks = []
    if files_to_process:
        st.info(f"📋 Processing {len(files_to_process)} new/changed PDF files...")
        progress_bar = st.progress(0)
        
        for i, (filename, filepath, file_hash) in enumerate(files_to_process):
            st.info(f"🔄 Processing: {filename}")
            
            raw_text = extract_text_from_pdf(filepath)
            if raw_text:
                chunks = splitter.split_text(raw_text)
                chunks = [f"(From {filename})\n\n{chunk}" for chunk in chunks]
                new_chunks.extend(chunks)
                
                # Update processed files log
                processed_files[filename] = file_hash
            
            progress_bar.progress((i + 1) / len(files_to_process))
    
    # Create or update vector store
    if new_chunks:
        with st.spinner("Creating/updating vector embeddings..."):
            if existing_vectorstore:
                # Add new embeddings to existing vectorstore
                st.info("🔄 Adding new embeddings to existing collection...")
                new_vectorstore = FAISS.from_texts(new_chunks, embedding=embedding_model)
                existing_vectorstore.merge_from(new_vectorstore)
                vectorstore = existing_vectorstore
                st.success(f"✅ Added {len(new_chunks)} new chunks to existing embeddings")
            else:
                # Create new vectorstore
                st.info("🆕 Creating new embedding collection...")
                vectorstore = FAISS.from_texts(new_chunks, embedding=embedding_model)
                st.success(f"✅ Created new embeddings with {len(new_chunks)} chunks")
            
            # Save updated vectorstore and processed files log
            vectorstore.save_local("faiss_index")
            save_processed_files_log(processed_files)
    else:
        vectorstore = existing_vectorstore
    
    # Show summary
    total_files = len(current_pdf_files)
    processed_count = len([f for f in current_pdf_files if f in processed_files])
    st.success(f"📊 Embedding Summary: {processed_count}/{total_files} PDF files processed")
    
    return vectorstore

def rebuild_all_embeddings(pdf_folder="pdfs"):
    """Force rebuild of all embeddings"""
    # Clear processed files log
    if os.path.exists("processed_files.json"):
        os.remove("processed_files.json")
    
    # Remove existing index
    if os.path.exists("faiss_index"):
        import shutil
        shutil.rmtree("faiss_index")
    
    st.info("🔄 Rebuilding all embeddings from scratch...")
    return process_pdfs_and_create_embeddings(pdf_folder, force_rebuild=True)

def load_existing_embeddings():
    """Load existing FAISS embeddings"""
    if os.path.exists("faiss_index"):
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectorstore = FAISS.load_local(
            "faiss_index",
            embeddings=embedding_model,
            allow_dangerous_deserialization=True
        )
        return vectorstore
    return None

def setup_local_llm():
    """Setup local LLM for RAG"""
    try:
        with st.spinner("Loading local LLM model..."):
            model_name = "google/flan-t5-base"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=512,
                temperature=0.7,
                do_sample=True,
                device_map="auto"
            )
            
            llm = HuggingFacePipeline(pipeline=pipe)
            return llm
    except Exception as e:
        st.error(f"Error loading local LLM: {str(e)}")
        return None

def create_qa_chain(vectorstore, llm):
    """Create QA chain for retrieval"""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

def get_realtime_information():
    """Get current date and time information"""
    current_date_time = datetime.now()
    day = current_date_time.strftime("%A")
    date = current_date_time.strftime("%d")
    month = current_date_time.strftime("%B")
    year = current_date_time.strftime("%Y")
    hour = current_date_time.strftime("%H")
    minute = current_date_time.strftime("%M")       
    second = current_date_time.strftime("%S")

    data = f"please use this realtime information if needed day:{day}, date:{date}, month: {month}, year: {year}, hour: {hour}, minutes:{minute}, seconds:{second}"
    return data

# Sidebar for model configuration
with st.sidebar:
    st.header("🤖 Model Configuration")
    
    # Model mode selection
    model_mode = st.radio(
        "Select Model Mode",
        ["Groq API (Cloud)", "Local RAG (PDF Documents)", "Hybrid Mode"]
    )
    
    # PDF Processing Section
    st.subheader("📄 PDF Document Processing")
    
    # Show current status
    if os.path.exists("faiss_index") and os.path.exists("processed_files.json"):
        processed_files = load_processed_files_log()
        current_pdfs = [f for f in os.listdir("pdfs") if f.endswith(".pdf")] if os.path.exists("pdfs") else []
        
        st.info(f"📊 Status: {len(processed_files)} files processed, {len(current_pdfs)} PDFs in folder")
        
        if processed_files:
            with st.expander("📋 View Processed Files"):
                for filename in processed_files.keys():
                    if filename in current_pdfs:
                        st.success(f"✅ {filename}")
                    else:
                        st.warning(f"⚠️ {filename} (file missing)")
    
    # Processing options
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Smart Update", type="primary", help="Only process new/changed PDFs"):
            vectorstore = process_pdfs_and_create_embeddings()
            if vectorstore:
                st.session_state.vectorstore = vectorstore
                st.success("✅ Embeddings updated!")
    
    with col2:
        if st.button("🔨 Rebuild All", help="Rebuild all embeddings from scratch"):
            vectorstore = rebuild_all_embeddings()
            if vectorstore:
                st.session_state.vectorstore = vectorstore
                st.success("✅ All embeddings rebuilt!")
    
    # Check for existing embeddings
    if st.button("Load Existing Embeddings"):
        vectorstore = load_existing_embeddings()
        if vectorstore:
            st.session_state.vectorstore = vectorstore
            st.success("Existing embeddings loaded!")
        else:
            st.warning("No existing embeddings found. Please process PDFs first.")
    
    # Local LLM Setup
    st.subheader("🧠 Local LLM Setup")
    
    if st.button("Setup Local LLM"):
        llm = setup_local_llm()
        if llm and st.session_state.vectorstore:
            st.session_state.qa_chain = create_qa_chain(st.session_state.vectorstore, llm)
            st.session_state.model_loaded = True
            st.success("Local RAG system ready!")
        elif not st.session_state.vectorstore:
            st.error("Please process PDFs first!")
    
    # Generation parameters
    st.subheader("⚙️ Generation Parameters")
    max_tokens = st.slider("Max Tokens", 50, 2000, 500)
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
    top_p = st.slider("Top P", 0.0, 1.0, 0.9, 0.05)
    
    # Model status
    st.subheader("📊 Status")
    if model_mode == "Groq API (Cloud)" and GroqAPIKey:
        st.success("✅ Groq API Ready")
    elif model_mode == "Local RAG (PDF Documents)" and st.session_state.qa_chain:
        st.success("✅ Local RAG Ready")
    elif model_mode == "Hybrid Mode" and GroqAPIKey and st.session_state.qa_chain:
        st.success("✅ Hybrid Mode Ready")
    else:
        st.warning("⚠️ Model Not Ready")
    
    # Embedding status
    if st.session_state.vectorstore:
        st.success("✅ Embeddings Loaded")
    else:
        st.info("ℹ️ No embeddings loaded")
    
    # Chat controls
    st.subheader("🎛️ Chat Controls")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    if st.button("Export Chat"):
        if st.session_state.messages:
            chat_data = {
                "timestamp": datetime.now().isoformat(),
                "messages": st.session_state.messages
            }
            st.download_button(
                "Download Chat JSON",
                json.dumps(chat_data, indent=2),
                file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

# Function to generate response
def generate_response(prompt, mode, max_tokens, temperature, top_p):
    """Generate response based on selected mode"""
    
    if mode == "Groq API (Cloud)":
        if not GroqAPIKey:
            return "Please set your Groq API key in the .env file."
        
        try:
            # System prompt
            system_prompt = f"""Hello, I am {Username}, You are a very accurate and advanced AI chatbot named {AssistantName} which also has real-time up-to-date information from the internet.
            *** Do not tell time until I ask, do not talk too much, just answer the question.***
            *** Reply in only English, even if the question is in Hindi, reply in English.***
            *** Do not provide notes in the output, just answer the question and never mention your training data. ***
            """
            
            system_chat_bot = [
                {"role": "system", "content": system_prompt}
            ]
            
            filtered_messages = []
            for msg in st.session_state.messages:
                if msg["role"] != "source":  # Skip source documents
                    filtered_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            completion = groq.chat.completions.create(
                model="llama3-8b-8192",
                messages=system_chat_bot + [{"role": "system", "content": get_realtime_information()}] + filtered_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=None,
                stream=True,
            )
            
            answer = ""
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    answer += chunk.choices[0].delta.content
            
            return answer.replace("</s>", "")
            
        except Exception as e:
            return f"Error with Groq API: {str(e)}"
    
    elif mode == "Local RAG (PDF Documents)":
        if not st.session_state.qa_chain:
            return "Please setup Local RAG system first by processing PDFs and setting up the local LLM."
        
        try:
            response = st.session_state.qa_chain.invoke({"query": prompt})
            answer = response["result"]
            
            # Add source documents to session state
            if response.get("source_documents"):
                source_info = "**Sources:**\n"
                for i, doc in enumerate(response["source_documents"]):
                    source_info += f"\n**Source {i+1}:**\n{doc.page_content[:200]}...\n"
                
                # Add source information as a separate message
                st.session_state.messages.append({
                    "role": "source",
                    "content": source_info,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
            
            return answer
            
        except Exception as e:
            return f"Error with Local RAG: {str(e)}"
    
    elif mode == "Hybrid Mode":
        # First try to get information from local documents
        if st.session_state.qa_chain:
            try:
                rag_response = st.session_state.qa_chain.invoke({"query": prompt})
                context_from_docs = rag_response["result"]
                
                # Then use Groq API with document context
                if GroqAPIKey:
                    system_prompt = f"""You are {AssistantName}, an advanced AI assistant. You have access to local document information and real-time capabilities.
                    
                    Context from local documents: {context_from_docs}
                    
                    Use this context along with your knowledge to provide comprehensive answers.
                    """
                    
                    completion = groq.chat.completions.create(
                        model="llama3-8b-8192",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "system", "content": get_realtime_information()},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        stream=True,
                    )
                    
                    answer = ""
                    for chunk in completion:
                        if chunk.choices[0].delta.content:
                            answer += chunk.choices[0].delta.content
                    
                    # Add source documents
                    if rag_response.get("source_documents"):
                        source_info = "**Sources from local documents:**\n"
                        for i, doc in enumerate(rag_response["source_documents"]):
                            source_info += f"\n**Source {i+1}:**\n{doc.page_content[:200]}...\n"
                        
                        st.session_state.messages.append({
                            "role": "source",
                            "content": source_info,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                    
                    return answer.replace("</s>", "")
                else:
                    return context_from_docs
                    
            except Exception as e:
                return f"Error in Hybrid Mode: {str(e)}"
        else:
            return "Please setup Local RAG system first for Hybrid Mode."
    
    return "Please select a valid model mode and ensure proper setup."

# Main chat interface
st.title("🤖 Custom Embedding Chatbot")
st.markdown(f"Chat with your AI assistant using **{model_mode}** mode")

# Display current mode info
if model_mode == "Local RAG (PDF Documents)":
    st.info("📄 This mode uses your local PDF documents for context-aware responses")
elif model_mode == "Hybrid Mode":
    st.info("🔄 This mode combines local PDF knowledge with cloud AI capabilities")
else:
    st.info("☁️ This mode uses cloud-based AI for general conversations")

# Display chat messages
chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <div class="message-avatar user-avatar">U</div>
                <div class="message-content">
                    <strong>You</strong>
                    <div>{message["content"]}</div>
                    <div class="timestamp">{message.get("timestamp", "")}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif message["role"] == "assistant":
            st.markdown(f"""
            <div class="chat-message bot-message">
                <div class="message-avatar bot-avatar">🤖</div>
                <div class="message-content">
                    <strong>{AssistantName}</strong>
                    <div>{message["content"]}</div>
                    <div class="timestamp">{message.get("timestamp", "")}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif message["role"] == "source":
            st.markdown(f"""
            <div class="source-doc">
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)

# Chat input
model_ready = (
    (model_mode == "Groq API (Cloud)" and GroqAPIKey) or
    (model_mode == "Local RAG (PDF Documents)" and st.session_state.qa_chain) or
    (model_mode == "Hybrid Mode" and GroqAPIKey and st.session_state.qa_chain)
)

if prompt := st.chat_input("Type your message here...", disabled=not model_ready):
    # Add user message to chat history
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.messages.append({
        "role": "user", 
        "content": prompt,
        "timestamp": timestamp
    })
    
    # Generate response
    with st.spinner("Generating response..."):
        response = generate_response(prompt, model_mode, max_tokens, temperature, top_p)
        
        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    # Rerun to update the chat display
    st.rerun()

# Instructions for users
with st.expander("📖 Setup & Usage Instructions", expanded=False):
    st.markdown("""
    ### 🚀 Quick Setup:
    
    1. **Create a `.env` file** with your API keys:
    ```
    GroqAPIKey=your_groq_api_key_here
    Username=YourName
    AssistantName=YourAssistant
    ```
    
    2. **For PDF Document Chat:**
       - Create a `pdfs` folder in your project directory
       - Add your PDF files to the folder
       - Click "🔄 Smart Update" to process only new/changed files
       - Or click "🔨 Rebuild All" to reprocess everything
       - Click "Setup Local LLM" to initialize the RAG system
    
    3. **Select your preferred mode:**
       - **Groq API (Cloud)**: Fast, general-purpose AI chat
       - **Local RAG (PDF Documents)**: Chat with your PDF documents
       - **Hybrid Mode**: Best of both worlds
    
    ### 🧠 Intelligent Embedding Management:
    
    - **Smart Updates**: Only processes new or modified PDF files
    - **File Change Detection**: Uses file size and modification time to detect changes
    - **Incremental Building**: Adds new embeddings to existing ones without rebuilding
    - **Processing Log**: Keeps track of processed files in `processed_files.json`
    - **Force Rebuild**: Option to rebuild all embeddings from scratch
    - **Status Display**: Shows which files are processed and up-to-date
    
    ### 📁 File Management:
    
    - **Add New PDFs**: Just drop new PDF files in the `pdfs` folder and click "Smart Update"
    - **Update Existing PDFs**: Modify files and they'll be automatically reprocessed
    - **Remove PDFs**: Deleted files are automatically removed from the processing log
    - **Bulk Operations**: Use "Rebuild All" for major changes or troubleshooting
    
    ### 📋 Requirements:
    
    Install these packages:
    ```bash
    pip install streamlit python-dotenv groq PyMuPDF
    pip install langchain langchain-community faiss-cpu
    pip install sentence-transformers transformers torch
    pip install langchain-huggingface
    ```
    
    ### 🎯 Features:
    
    - **Multi-mode Operation**: Switch between cloud AI and local document processing
    - **PDF Processing**: Extract and embed content from PDF documents
    - **Retrieval-Augmented Generation (RAG)**: Get accurate answers from your documents
    - **Source Citations**: See which documents were used for answers
    - **Real-time Information**: Access to current date/time
    - **Chat Export**: Save your conversations
    - **Responsive UI**: Adapts to light/dark themes
    
    ### 🔧 Customization:
    
    - Modify the `model_name` in `setup_local_llm()` to use different local models
    - Adjust chunking parameters in `process_pdfs_and_create_embeddings()`
    - Customize system prompts for different use cases
    - Add support for other document formats (DOCX, TXT, etc.)
    """)

# Footer
st.markdown("---")
st.markdown("🚀 **Custom Embedding Chatbot** - Combining cloud AI with local document intelligence!")