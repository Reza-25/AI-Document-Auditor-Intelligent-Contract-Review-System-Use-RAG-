# 🔍 AI Document Auditor: Intelligent Contract Review System

### 🚀 Overview
**AI Document Auditor** is a high-performance RAG (Retrieval-Augmented Generation) system designed to automate the auditing of complex legal documents, contracts, and SOPs. Instead of manually reading hundreds of pages, users can instantly identify high-risk clauses, penalties, and obligations through an AI-powered interface.

### 💡 The Problem
Reviewing legal documents like the **Apple Developer Program License Agreement** is time-consuming and prone to human error. Critical details regarding fines, termination, or jurisdiction are often buried in dense text, posing significant operational risks for businesses and individuals.

### ✨ Key Features
- **⚡ Ultra-Fast Auditing:** Powered by **Groq (Llama 3.3)** for near-instant response generation (under 2 seconds).
- **🛡️ Zero-Hallucination RAG:** Uses **Gemini Embeddings** and **ChromaDB** to ensure answers are strictly based on the uploaded document.
- **📋 Automated Audit Report:** One-click execution of a comprehensive audit checklist covering Fines, Confidentiality, Termination, and High-Risk clauses.
- **📍 Precise Citations:** Every answer includes exact page references for easy verification.
- **💡 Smart Suggestions:** Automatically generates context-aware questions based on the specific content of your document.

### 🛠️ Tech Stack
- **Framework:** LangChain
- **Frontend:** Streamlit
- **LLM Engine:** Llama 3.3-70b via Groq
- **Vector Search:** Google Gemini Embedding-001
- **Database:** ChromaDB (Local Vector Store)

### 📊 How It Works
1. **Ingestion:** The PDF is extracted per page and split into semantic chunks.
2. **Vectorization:** Chunks are converted into high-dimensional vectors using Gemini and stored in ChromaDB.
3. **Retrieval:** When a question is asked, the system retrieves the 4 most relevant chunks from the document.
4. **Synthesis:** The LLM synthesizes a cited answer based *only* on the retrieved context.

### 📸 Screenshots
<img width="959" height="588" alt="Screenshot 2026-04-26 121207" src="https://github.com/user-attachments/assets/3a2f90fe-deb1-4389-a8d7-27d8acd24019" />
<img width="1591" height="879" alt="Screenshot 2026-04-28 185343" src="https://github.com/user-attachments/assets/ea0692c2-5464-45d3-9bf8-79277c9c28c8" />
<img width="1919" height="862" alt="Screenshot 2026-04-28 185825" src="https://github.com/user-attachments/assets/89609680-f819-42b4-8148-03f0b5e0dd0a" />
### 🔧 Installation
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Add your API keys to a `.env` file
4. Run the app: `streamlit run app.py`

---
**Developed by:** [Reza]  
*Turning complex data into actionable business intelligence.*
