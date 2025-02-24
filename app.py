import streamlit as st
import faiss
import openai
import numpy as np
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

if not openai.api_key:
    raise ValueError("OpenAI API key not found!")

# Load FAISS index
index = faiss.read_index("vector_store.index")

# Load chunked text data
with open("chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

def get_embedding(text):
    """Generate embeddings for input text."""
    response = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

def get_relevant_chunks(query, k=3):
    """Retrieve the most relevant chunks from the FAISS index."""
    query_embedding = np.array([get_embedding(query)], dtype='float32')
    D, I = index.search(query_embedding, k)
    return [chunks[i] for i in I[0]]

# Streamlit app UI
st.title("Precomputed FAISS Course Assistant")
query = st.text_input("Ask a question about the course:")

if query:
    relevant_chunks = get_relevant_chunks(query)
    st.write("Most relevant information:")
    for chunk in relevant_chunks:
        st.markdown(f"```\n{chunk}\n```")