# app.py

import streamlit as st
import openai
import faiss
import time
import numpy as np
from dotenv import load_dotenv
import os
import json

# Load environment variables
load_dotenv()

class CourseBot:
    def __init__(self):
        """Initialize the RAG chatbot with necessary components."""
        openai.api_key = os.getenv('OPENAI_API_KEY')
        if not openai.api_key:
            raise ValueError("OpenAI API key not found in environment variables!")
            
        self.model = os.getenv('MODEL_NAME', 'gpt-4')
        self.temperature = float(os.getenv('TEMPERATURE', '0.7'))
        self.dimension = 1536  # OpenAI embedding dimension
        self.initialize_vector_store()
        
    def load_content(self, filename):
        """Load content from text file."""
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                return file.read()
        except FileNotFoundError:
            st.error(f"Could not find {filename}. Please ensure all content files are present.")
            return ""

    def get_embedding(self, text):
        """Get embeddings from OpenAI API."""
        response = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding

    def initialize_vector_store(self):
        """Create FAISS vector store from course content."""
        # Load content from files
        syllabus = self.load_content('syllabus.txt')
        piazza = self.load_content('piazza.txt')
        course_notes = self.load_content('course_notes.txt')
        
        # Combine and split content into chunks
        all_content = f"{syllabus}\n\n{piazza}\n\n{course_notes}"
        self.chunks = [chunk for chunk in all_content.split('\n\n') if chunk.strip()]
        
        # Create embeddings for chunks
        embeddings = [self.get_embedding(chunk) for chunk in self.chunks]
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(np.array(embeddings, dtype='float32'))

    def get_relevant_chunks(self, query, k=3):
        """Get most relevant chunks for a query."""
        query_embedding = self.get_embedding(query)
        D, I = self.index.search(np.array([query_embedding], dtype='float32'), k)
        return [self.chunks[i] for i in I[0]]

    def get_response(self, query):
        """Get response for user query with source context."""
        # Get relevant context
        relevant_chunks = self.get_relevant_chunks(query)
        context = "\n\n".join(relevant_chunks)
        
        # Prepare messages for chat completion
        messages = [
            {"role": "system", "content": "You are a helpful course assistant. Use the provided context to answer questions accurately."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
        
        # Get response from OpenAI
        response = openai.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature
        )
        
        return {
            'answer': response.choices[0].message.content,
            'source_docs': relevant_chunks
        }

def main():
    st.title("CS 101 Course Assistant")
    st.write("Ask any questions about the Python Programming course!")

    # Check for API key in environment variables
    if not os.getenv('OPENAI_API_KEY'):
        st.error("OpenAI API key not found! Please set it in your .env file.")
        st.stop()

    # Initialize session state with loading indicators
    if 'chatbot' not in st.session_state:
        init_placeholder = st.empty()
        
        try:
            # Show initialization progress
            with init_placeholder.container():
                st.markdown("### 🚀 Initializing Course Assistant...")
                
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Loading environment
                status_text.text("📚 Loading course materials...")
                progress_bar.progress(25)
                
                # Initializing embeddings
                status_text.text("🔄 Initializing embeddings and vector store...")
                progress_bar.progress(50)
                
                # Create chatbot instance
                st.session_state.chatbot = CourseBot()
                status_text.text("✨ Setting up chat interface...")
                progress_bar.progress(75)
                
                # Initialize messages
                st.session_state.messages = []
                status_text.text("✅ System ready!")
                progress_bar.progress(100)
                
                # Success message
                st.success("Initialization complete! You can now ask questions about the course.")
                
            # Clear the initialization messages after a brief delay
            time.sleep(1)
            init_placeholder.empty()
            
        except Exception as e:
            st.error(f"Error initializing chatbot: {str(e)}")
            st.stop()

    # Sidebar for information
    with st.sidebar:
        st.write("System Information:")
        st.write(f"- Using {os.getenv('MODEL_NAME', 'gpt-4')} model")
        st.write("- FAISS vector store")
        st.write("- Context-aware responses")

        # Add syllabus download section at the bottom
        st.markdown("---")  # Separator line
        try:
            with open('syllabus.txt', 'r', encoding='utf-8') as file:
                syllabus_content = file.read()
            st.download_button(
                label="📄 Download Syllabus",
                data=syllabus_content,
                file_name="CS101_Syllabus.txt",
                mime="text/plain",
                help="Click to download the course syllabus"
            )
        except FileNotFoundError:
            st.error("Syllabus file not found")
    
    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Query input
    query = st.chat_input("Ask a question about the course...")
    
    if query:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Get bot response
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            try:
                # Create a placeholder for the loading animation
                with st.spinner("🤔 Let me think about that..."):
                    response = st.session_state.chatbot.get_response(query)
                    
                # Display the response
                st.write(response['answer'])
                
                # Show source context in expander
                with st.expander("View Source Context"):
                    for doc in response['source_docs']:
                        st.markdown(f"```\n{doc}\n```")
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response['answer']})
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

        # Feedback buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Helpful"):
                st.success("Thank you for your feedback!")
        with col2:
            if st.button("👎 Not Helpful"):
                feedback = st.text_area("What could be improved?")
                if st.button("Submit Feedback"):
                    st.error("Thanks for letting us know!")

if __name__ == "__main__":
    main()