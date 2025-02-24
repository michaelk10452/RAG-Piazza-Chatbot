# 🎓 RAG-Piazza Chatbot

## 📌 System Overview
The **RAG-Piazza Chatbot** is a **Retrieval-Augmented Generation (RAG)**-based course assistant designed to enhance student learning through **efficient and intelligent information retrieval**. Built using **Streamlit, FAISS, and OpenAI GPT**, the chatbot enables students to interact with their course material, syllabus, and Piazza discussions dynamically, retrieving the most relevant information before generating accurate responses.

This system was originally built for **CS 101** and comes with a preloaded dataset that has been vectorized and chunked for optimal performance. However, it can be easily modified to work with any course that has Piazza post data, a syllabus, and other course materials.

## 🚀 Why This System Stands Out
- **Optimized for Speed**: Precomputed FAISS embeddings ensure near-instantaneous retrieval of relevant course materials.
- **Scalability & Efficiency**: Designed with **vector databases (FAISS)** to handle large-scale information retrieval tasks with minimal latency.
- **Context-Aware Responses**: Unlike traditional chatbots, it combines document retrieval with GPT-generated responses, ensuring answers are grounded in **relevant course materials**.
- **User-Centric Interface**: Built with **Streamlit**, offering a sleek, easy-to-use chat interface for students and educators.
- **Secure API Key Handling**: Uses `.env` files and Streamlit Secrets to **protect sensitive credentials** and ensure safe API usage.

## 🧠 Why RAG is Critical for Educational Assistants
RAG (Retrieval-Augmented Generation) is particularly valuable for educational applications like this course assistant:

- **Knowledge Accuracy**: By grounding responses in verified course materials, RAG prevents hallucination and ensures information accuracy—critical for educational contexts where factual correctness is non-negotiable.
- **Up-to-Date Information**: Easily incorporate course-specific content including the latest lectures, Piazza discussions, and instructor announcements without retraining the entire model.
- **Course-Specific Context**: Captures nuances of how specific concepts are taught in CS 101, respecting the instructor's terminology and pedagogical approach.
- **Cited Sources**: Enables the system to reference specific materials, allowing students to verify information and explore topics more deeply in the original course documents.
- **Reduced Bias**: Minimizes inherent biases in foundation models by prioritizing retrieving actual course content over generating responses from pre-trained knowledge.

## 📖 Example Queries & Responses
#### 📌 **Course-Specific Retrieval Questions**
> *"What topics are covered on the midterm exam?"*
<img width="1500" alt="Screenshot 2025-02-24 at 3 36 30 PM" src="https://github.com/user-attachments/assets/2f869cf9-f839-49c5-aa59-74a686b681d7" />

#### 📌 **Assignment-Specific Questions**
> *"How do I resolve a Git merge conflict?"*
<img width="1501" alt="Screenshot 2025-02-24 at 3 37 49 PM" src="https://github.com/user-attachments/assets/1fc5bfb2-4470-4048-b80d-e21f4ee4f37f" />

#### 📌 **Course Policy & Exam Guidelines Questions**
> *"Can I bring a cheat sheet to the final exam?"*
<img width="1500" alt="Screenshot 2025-02-24 at 3 39 01 PM" src="https://github.com/user-attachments/assets/ff3f8714-d523-4516-845b-9df94d1a37e2" />



Built by Michael Kurdahi

This project is open for contributions! If you have ideas for improvements or want to adapt it for other courses, please feel free to fork the repository and submit a pull request. Together, we can make learning more accessible and interactive for students everywhere.
