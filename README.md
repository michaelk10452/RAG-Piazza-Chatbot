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
#### 📌 **Topic-Specific Questions**
> *"What are the key topics covered in the CS 101 syllabus?"*

![Topic-Specific Question](https://user-images.githubusercontent.com/example/topic-question.png)

#### 📌 **Course-Specific Questions**
> *"Explain the difference between lists and tuples in Python with examples."*

![Course-Specific Question](https://user-images.githubusercontent.com/example/course-question.png)

Built by Michael Kurdahi

This project is open for contributions! If you have ideas for improvements or want to adapt it for other courses, please feel free to fork the repository and submit a pull request. Together, we can make learning more accessible and interactive for students everywhere.