from bs4 import BeautifulSoup
import requests
import os
import pickle
import time
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Configuration
LLM_API_KEY = "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"
LLM_MODEL_NAME = "llama-3.1-70b-versatile"
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_STORE_FILE = "faiss_store_openai.pkl"

# Initializing LLM
def initialize_llm():
    return ChatGroq(temperature=0, groq_api_key=LLM_API_KEY, model_name=LLM_MODEL_NAME)

# Function to scrape content from a website
def scrape_website(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        return ' '.join([para.get_text() for para in paragraphs])
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return ""

# Processing text into chunks and save FAISS index
def process_text_and_save(text, file_path):
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text_chunks = text_splitter.split_text(text)

    print("Creating embeddings and building FAISS index...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
    vectorstore = FAISS.from_texts(text_chunks, embeddings)

    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)

    print(f"FAISS index saved to {file_path}.")

# Main function to process websites and save embeddings
def process_websites(urls):
    print("Starting website processing...")
    all_text = "\n".join([scrape_website(url.strip()) for url in urls])
    process_text_and_save(all_text, FAISS_STORE_FILE)

# Function to query the FAISS store
def query_faiss_store(query, file_path, llm):
    if not os.path.exists(file_path):
        print(f"FAISS index file not found at {file_path}. Please process websites first.")
        return

    with open(file_path, "rb") as f:
        vectorstore = pickle.load(f)

    retriever = vectorstore.as_retriever()
    chain = RetrievalQA.from_llm(llm=llm, retriever=retriever)
    return chain.run(query)

# Main script
if __name__ == "__main__":
    # Initialize LLM
    llm = initialize_llm()

    # Get URLs from the user
    urls = input("Enter the website URLs (comma-separated): ").split(',')

    # Process the URLs and build the FAISS index
    process_websites(urls)

    # Quering the index
    while True:
        query = input("Ask a Question (or type 'exit' to quit): ")
        if query.lower() == "exit":
            print("Exiting program.")
            break
        result = query_faiss_store(query, FAISS_STORE_FILE, llm)
        if result:
            print("Answer:")
            print(result)
