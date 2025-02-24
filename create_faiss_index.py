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

# OpenAI embedding function
def get_embedding(text):
    response = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

# Load course materials
def load_content(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Could not find {filename}")
        return ""

# Load all content
syllabus = load_content('syllabus.txt')
piazza = load_content('piazza.txt')
course_notes = load_content('course_notes.txt')

# Split content into chunks
all_content = f"{syllabus}\n\n{piazza}\n\n{course_notes}"
chunks = [chunk for chunk in all_content.split('\n\n') if chunk.strip()]

# Create embeddings
embeddings = np.array([get_embedding(chunk) for chunk in chunks], dtype='float32')

# Initialize FAISS index
dimension = 1536  # OpenAI embedding dimension
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save FAISS index
faiss.write_index(index, "vector_store.index")

# Save chunks for later retrieval
with open("chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f)

print("FAISS index and chunk data saved!")