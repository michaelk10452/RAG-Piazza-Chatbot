import streamlit as st
import openai
import faiss
import time
import numpy as np
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class CourseBot:
    def __init__(self):
        """Initialize the RAG chatbot with precomputed vectors."""
        openai.api_key = os.getenv('OPENAI_API_KEY')
        if not openai.api_key:
            raise ValueError("OpenAI API key not found in environment variables!")
            
        self.model = os.getenv('MODEL_NAME', 'gpt-4')
        self.temperature = float(os.getenv('TEMPERATURE', '0.7'))
        self.dimension = 1536  # OpenAI embedding dimension
        
        # Load precomputed vectors
        self.load_precomputed_vectors()

    def load_precomputed_vectors(self):
        """Load precomputed FAISS index and chunks."""
        try:
            # Load FAISS index
            self.index = faiss.read_index("vector_store.index")
            
            # Load chunks
            with open("chunks.json", "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
                
        except FileNotFoundError:
            raise ValueError("Precomputed vectors not found! Please run the vector creation script first.")

    def get_embedding(self, text):
        """Get embeddings from OpenAI API."""
        response = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding

    def get_relevant_chunks(self, query, k=3):
        """Get most relevant chunks for a query using precomputed vectors."""
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
            {"role": "system", "content": """You are a knowledgeable teaching assistant for this specific computer science course. Your responses should:
            1. Always provide specific information from the course materials when available
            2. Look for explicit course details in the provided context
            3. Never say 'the context does not provide' - instead, look harder in the context for relevant information
            4. If truly unsure about specific details, say 'Based on the available course materials, this appears to be a computer science course focused on [relevant topics found in context]'
            5. Use a helpful, informative tone like a real TA would
            
            Remember that students are asking about THIS specific course, so avoid generic answers."""},
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
                
                # Loading vectors
                status_text.text("🔄 Loading precomputed vectors...")
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
        st.write("- Using precomputed FAISS vectors")
        st.write("- Context-aware responses from course materials")
        st.write("- Feedback buttons for user experience improvement")
        st.markdown("---")
        st.write("Course Materials:")
        st.write("- Syllabus")
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
        st.write("- Piazza posts")
        if st.button("📋 See Piazza Posts"):
            try:
                with open('piazza.txt', 'r', encoding='utf-8') as file:
                    piazza_content = file.read()
                st.markdown("### 📋 Piazza Posts")
                st.text_area("Recent Posts", value=piazza_content, height=300, disabled=True)
            except FileNotFoundError:
                st.error("Piazza posts file not found")
        st.write("- Course notes")
        st.markdown("---")
        st.write("Feedback:")
        st.write("If you find this assistant helpful, please let us know!")
        st.markdown("---")
        st.write("Developed by Michael Kurdahi")
        st.write("Last updated: February 2025")

        # Add syllabus download section
        st.markdown("---")
            
        # Add Piazza posts section
        st.markdown("---")
    
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
                        st.markdown(f"\n{doc}\n")
                
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