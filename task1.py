import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai

# Step 1: PDF Parsing and Text Extraction
def extract_pdf_content(file_path):
    content = []
    with pdfplumber.open(file_path) as pdf:
        for page_number, page in enumerate(pdf.pages):
            text = page.extract_text()
            tables = page.extract_tables()
            content.append({"page": page_number + 1, "text": text, "tables": tables})
    return content

# Step 2: Chunking and Embedding
def generate_embeddings(data):
    chunks = []
    chunk_map = {}  # Map to track chunks and their source
    embeddings = []
    model = SentenceTransformer('all-MiniLM-L6-v2')

    for page in data:
        text = page["text"]
        if text:
            page_chunks = text.split("\n\n")  # Chunking based on paragraphs
            chunks.extend(page_chunks)
            for chunk in page_chunks:
                chunk_map[len(chunks) - 1] = f"Page {page['page']}"
                embeddings.append(model.encode(chunk))

    return chunks, np.array(embeddings), chunk_map

# Step 3: Store Embeddings in FAISS
def create_faiss_index(embeddings):
    d = embeddings.shape[1]  # Dimensionality of the embedding
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index

# Step 4: Query Handling and Retrieval
def search_query(query, model, index, chunks, chunk_map, top_k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    results = [{"chunk": chunks[i], "source": chunk_map[i]} for i in indices[0]]
    return results

# Step 5: Integration with LLM for Response Generation
def generate_response(query, retrieved_chunks):
    openai.api_key = "your_openai_api_key"  # Replace with your OpenAI API Key
    context = "\n\n".join([f"From {item['source']}:\n{item['chunk']}" for item in retrieved_chunks])
    prompt = f"Based on the following information:\n\n{context}\n\nAnswer the query: {query}"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']

# Main Workflow
if __name__ == "__main__":
    # Input PDF Path
    pdf_path = "tables-charts-and-graphs-with-examples-from.pdf"  # Replace with your file path

    # Step 1: Extract content from PDF
    content = extract_pdf_content(pdf_path)

    # Step 2: Generate embeddings and create FAISS index
    chunks, embeddings, chunk_map = generate_embeddings(content)
    faiss_index = create_faiss_index(embeddings)

    # Initialize embedding model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Step 3: User Query Examples
    query_1 = "What is the unemployment rate for people with a bachelor's degree?"
    query_2 = "Explain the tabular data on page 6."

    # Search and retrieve results
    results_1 = search_query(query_1, embedding_model, faiss_index, chunks, chunk_map)
    results_2 = search_query(query_2, embedding_model, faiss_index, chunks, chunk_map)

    # Generate responses using OpenAI LLM
    response_1 = generate_response(query_1, results_1)
    response_2 = generate_response(query_2, results_2)

    # Output the results
    print("Query 1 Response:")
    print(response_1)
    print("\nQuery 2 Response:")
    print(response_2)
