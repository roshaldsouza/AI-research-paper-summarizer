📄 GenAI Research Paper Summarizer & Q&A (RAG-based)
🚀 Project Overview

This project is a Generative AI–based Research Paper Summarizer and Question-Answering system built using LLaMA and Retrieval-Augmented Generation (RAG). It allows users to upload or provide research paper content and then ask natural language questions such as “What are the key findings?” or “Summarize the paper”.
The system retrieves relevant sections from the document and generates accurate, context-aware answers.

🧠 Key Features

📑 Research paper summarization

❓ Question answering from documents

🔍 Context-aware responses using RAG

⚡ Local LLaMA model (no paid APIs)

🛡️ Reduced hallucinations by grounding answers in retrieved content

🧪 Simple CLI-based interaction for easy testing

🏗️ Architecture (RAG Pipeline)

Document Loading – Research paper text is loaded and preprocessed

Chunking – Text is split into smaller semantic chunks

Embedding – Chunks are converted into vector embeddings

Vector Store – Stored for similarity-based retrieval

Retriever – Fetches the most relevant chunks for a user query

LLaMA Model – Generates the final answer using retrieved context

🛠️ Tech Stack

Python

LLaMA (via Ollama)

Retrieval-Augmented Generation (RAG)

Vector Database (e.g., FAISS / Chroma)

CLI-based Interface

📂 Project Structure
.
├── paper_summarizer.py      # Main RAG pipeline
├── test_summarizer.py       # Testing and sample queries
├── data/                    # Research papers / documents
├── embeddings/              # Vector store
├── requirements.txt
└── README.md

▶️ How to Run the Project
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Start LLaMA using Ollama
ollama run llama3

3️⃣ Run the Application
python paper_summarizer.py

💡 Sample Questions

What are the key findings of this paper?

Give a summary of the research

What problem does the paper solve?

What methods were used?

🎯 Use Cases

Students reading academic papers
Researchers needing quick insights
Literature review assistance
AI-powered document understanding systems

🌱 Future Improvements

Web UI using React or Streamlit
Support for PDF uploads
Multi-document comparison
Improved chunk ranking
Citation-based answers

🧑‍💻 Author

Roshal Dsouza
Computer Science Student | Full Stack & GenAI Developer
