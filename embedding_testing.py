# After running the pip commands above, use this code:

from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

# Use different import and configuration to avoid the StopIteration bug
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

print("Loading model...")

# Load model locally to avoid API issues
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Create pipeline
pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    temperature=0.7,
    do_sample=True,
    device_map="auto"  # Use GPU if available
)

# Create LLM
llm = HuggingFacePipeline(pipeline=pipe)

# Your original embedding setup (with fixed model name)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Your original question
question = "aba taba kuba"
print(f"Processing: {question}")

response = qa_chain.invoke({"query": question})
print("Answer:", response["result"])

for i, doc in enumerate(response["source_documents"]):
    print(f"\nSource {i+1}:\n{doc.page_content}")