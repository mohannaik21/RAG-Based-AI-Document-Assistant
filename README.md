# 🤖RAG Based AI Chatbot - AI Document Assistant

Project Explanation:
This AI-Powered Document Chat Assistant is a sophisticated question-answering system that enables users to upload documents (PDF, DOCX, TXT) and engage in intelligent conversations about their content. The application processes documents by extracting text, cleaning and preprocessing it, then splitting it into semantic chunks for efficient retrieval. It utilizes advanced Retrieval-Augmented Generation (RAG) architecture, combining vector embeddings with traditional keyword search for hybrid document retrieval. When users ask questions, the system searches through the document chunks using both semantic similarity (via sentence transformers) and keyword matching, retrieves the most relevant context, and generates comprehensive answers using Mistral AI's language model. The interface features a clean, single-section design with real-time chat functionality, confidence scoring for responses, and automatic source attribution, making it ideal for document analysis, research assistance, and content exploration.
Technical Stack & Keywords
Frontend & UI

Streamlit - Python web framework for rapid UI development, handles user interactions and real-time updates
HTML/CSS - Custom styling for chat bubbles, animations, and responsive design elements
JavaScript (implicit) - Streamlit's built-in reactivity for dynamic content updates

Document Processing

PyPDF2 - PDF text extraction library, handles multi-page document parsing and content retrieval
python-docx - Microsoft Word document processing, extracts text from DOCX files including tables
Text Preprocessing - Regular expressions (re module) for cleaning, normalizing, and structuring raw text
Chunking Algorithm - Overlapping text segmentation with sentence boundary detection for context preservation

AI & Machine Learning

Sentence Transformers - Neural network models for semantic text embeddings, converts text to vector representations
all-MiniLM-L6-v2 - Lightweight transformer model optimized for sentence similarity tasks
Mistral AI API - Large language model for natural language generation and question answering
RAG (Retrieval-Augmented Generation) - Architecture combining information retrieval with generative AI

Vector Search & Indexing

FAISS - Facebook's similarity search library for efficient nearest neighbor retrieval in high-dimensional spaces
Scikit-learn - Machine learning library providing TF-IDF vectorization and cosine similarity calculations
TF-IDF (Term Frequency-Inverse Document Frequency) - Statistical measure for keyword-based document relevance
Hybrid Search - Combines semantic embeddings with traditional keyword matching for improved accuracy

Data Management

NumPy - Numerical computing library for vector operations and mathematical computations
Session State Management - Streamlit's persistent storage for maintaining chat history and document data
Metadata Tracking - Document source attribution, chunk indexing, and confidence scoring

APIs & Communication

Requests - HTTP library for API communication with Mistral AI services
python-dotenv - Environment variable management for secure API key storage
Error Handling - Retry logic, timeout management, and user-friendly error messages

Architecture Patterns

Modular Design - Separated concerns with distinct files for document loading, vector storage, and QA processing
Real-time Processing - Streaming document upload with progress indicators and immediate feedback
Confidence Scoring - Similarity score aggregation for response reliability assessment










# 🤖 RAG-Based AI Document Assistant

📌 Project Overview

This **AI-Powered Document Chat Assistant** is a Retrieval-Augmented Generation (RAG) based system that allows users to upload documents (PDF, DOCX, TXT) and ask intelligent questions about their content.

The application processes documents by extracting text, cleaning and preprocessing it, splitting it into semantic chunks, and storing them in a vector database. When a user asks a question, the system retrieves the most relevant document chunks using **hybrid search** (semantic + keyword-based) and generates accurate answers using **Mistral AI’s Large Language Model**.

The UI is built with **Streamlit**, offering a clean chat interface with:
- Real-time responses
- Confidence scoring
- Automatic source attribution

This makes the project ideal for **document analysis, research assistance, and content exploration**.


## 🧠 How It Works (RAG Architecture)
1. User uploads a document (PDF, DOCX, or TXT)
2. Text is extracted and preprocessed
3. Content is split into overlapping semantic chunks
4. Chunks are converted into embeddings using Sentence Transformers
5. Embeddings are stored in a FAISS vector database
6. User queries are embedded and matched against stored chunks
7. Relevant context is passed to Mistral LLM for answer generation
8. Final response is shown with confidence score and source reference


## 🛠️ Technical Stack & Keywords

### 🔹 Frontend & UI
- **Streamlit** – Python web framework for rapid UI development
- **HTML/CSS** – Custom styling for chat bubbles and layout
- **JavaScript (implicit)** – Streamlit’s reactive rendering


### 🔹 Document Processing
- **PyPDF2** – PDF text extraction
- **python-docx** – DOCX document parsing
- **Regex (re module)** – Text cleaning and normalization
- **Chunking Algorithm** – Overlapping text chunks with sentence boundary detection


### 🔹 AI & Machine Learning
- **Sentence Transformers** – Semantic text embeddings
- **all-MiniLM-L6-v2** – Lightweight embedding model
- **Mistral AI API** – Large Language Model for answer generation
- **RAG (Retrieval-Augmented Generation)** – Combines retrieval with generative AI


### 🔹 Vector Search & Retrieval
- **FAISS** – Efficient similarity search in high-dimensional space
- **Scikit-learn** – TF-IDF vectorization & cosine similarity
- **TF-IDF** – Keyword-based relevance scoring
- **Hybrid Search** – Semantic + keyword-based retrieval


### 🔹 Data & State Management
- **NumPy** – Vector operations
- **Streamlit Session State** – Chat history persistence
- **Metadata Tracking** – Source attribution and confidence scoring


### 🔹 APIs & Utilities
- **Requests** – API calls to Mistral AI
- **python-dotenv** – Secure environment variable handling
- **Error Handling** – Retry logic and graceful failures


### 🔹 Architecture Patterns
- **Modular Design** – Separate files for document loading, vector storage, and QA logic
- **Real-time Processing** – Immediate feedback during uploads
- **Confidence Scoring** – Similarity-based response reliability


## 📂 Project Structure
├── app.py

├── document_loader.py

├── qa_engine.py

├── vector_store.py

├── utils.py

├── requirements.txt

├── README.md

├── .gitignore

└── venv/ (ignored)

## ⚙️ Installation & Setup (Step-by-Step)
### 1️⃣ Clone the repository
git clone https://github.com/mohannaik21/RAG-Based-AI-Document-Assistant.git

cd RAG-Based-AI-Document-Assistant

### 2️⃣ Create a virtual environment
python -m venv venv

Activate it (Windows):
venv\Scripts\activate

### 3️⃣ Install dependencies
pip install --upgrade pip

pip install -r requirements.txt

### 4️⃣ Configure environment variables (IMPORTANT)
Create a file named .env in the project root directory:

MISTRAL_API_KEY=your_mistral_api_key_here

⚠️ Notes:
Ensure the file name is exactly .env (not .env.txt)

.env is excluded from GitHub using .gitignore

Restart the app after creating .env


### 5️⃣ Run the application
streamlit run app.py

Open in browser:
http://localhost:8501




# To download requiremnts
pip install -r requirements.txt

# To run
streamlit run app.py"# RAG-Based-AI-Document-Assistant" 
