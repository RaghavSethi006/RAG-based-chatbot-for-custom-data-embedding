import os
import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


#load embedding Model
embedding_model = HuggingFaceEmbeddings(model_name="all-miniLM-L6-v2")

#setup chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n","\n","."," ",""]
)

#Directory Containing PDFs
pdf_folder = "pdfs"
all_chunks = []

def extract_text_from_pdf(path):
    doc = fitz.open(path)
    text = "\n".join(page.get_text() for page in doc)
    return text

#loop trough all pdfs
for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        filepath=os.path.join(pdf_folder,filename)
        raw_text = extract_text_from_pdf(filepath)
        chunks = splitter.split_text(raw_text)
        chunks = [f"(From {filename})\n\n{chunk}" for chunk in chunks]
        all_chunks.extend(chunks)

#embed and save
vectorstore = FAISS.from_texts(all_chunks,embedding=embedding_model)
vectorstore.save_local("faiss_index")