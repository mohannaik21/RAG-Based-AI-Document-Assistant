import streamlit as st
import time
from datetime import datetime
from document_loader import load_and_split_document
from vector_store import VectorStore
from qa_engine import generate_answer

# Page configuration
st.set_page_config(
    page_title="📚 Document Chat",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Simple, clean CSS
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
        max-width: 800px;
        margin: 0 auto;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 10px;
        animation: fadeIn 0.3s ease-in;
    }
    
    .user-message {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        margin-left: 10%;
    }
    
    .bot-message {
        background: #f3e5f5;
        border-left: 4px solid #9c27b0;
        margin-right: 10%;
    }
    
    .upload-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
        border: 2px dashed #dee2e6;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .success-box {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .typing {
        background: #f1f3f4;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 0.7; }
        50% { opacity: 1; }
    }
    
    .error-message {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Hide streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    if "vs" not in st.session_state:
        st.session_state.vs = None
    if "history" not in st.session_state:
        st.session_state.history = []
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []

def main():
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="color: #2c3e50; margin-bottom: 0.5rem;">📚 Document Chat Assistant</h1>
        <p style="color: #7f8c8d; font-size: 1.1rem;">Upload your documents and ask questions about them</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload section (always at top)
    with st.container():
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "📄 Choose your documents",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            help="Supported formats: PDF, Word documents, Text files"
        )
        
        if uploaded_files:
            process_uploaded_files(uploaded_files)
        
        # Show upload status
        if st.session_state.uploaded_files:
            st.markdown(f"""
            <div class="success-box">
                ✅ {len(st.session_state.uploaded_files)} document(s) loaded and ready for questions!
            </div>
            """, unsafe_allow_html=True)
            
            # Clear documents button
            if st.button("🗑️ Clear All Documents", key="clear_docs"):
                st.session_state.vs = None
                st.session_state.uploaded_files = []
                st.session_state.history = []
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content area
    if st.session_state.vs:
        render_chat_interface()
    else:
        # Welcome message when no documents
        if not st.session_state.uploaded_files:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; color: #6c757d;">
                <h3>👆 Upload your documents above to get started</h3>
                <p>Supported formats: PDF, DOCX, TXT</p>
                <p>Then you can ask questions about your content!</p>
            </div>
            """, unsafe_allow_html=True)

def process_uploaded_files(uploaded_files):
    """Process uploaded files"""
    new_files = []
    
    # Check for new files
    for uploaded_file in uploaded_files:
        if not any(f['name'] == uploaded_file.name for f in st.session_state.uploaded_files):
            new_files.append(uploaded_file)
    
    if new_files:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, uploaded_file in enumerate(new_files):
            status_text.text(f"Processing {uploaded_file.name}...")
            progress_bar.progress((i + 1) / len(new_files))
            
            try:
                # Load and split document
                chunks = load_and_split_document(uploaded_file)
                
                # Create or update vector store
                if st.session_state.vs is None:
                    st.session_state.vs = VectorStore()
                
                st.session_state.vs.add_documents(chunks, uploaded_file.name)
                
                # Store file info
                file_info = {
                    "name": uploaded_file.name,
                    "chunks": len(chunks),
                    "uploaded_at": datetime.now().isoformat()
                }
                
                st.session_state.uploaded_files.append(file_info)
                
            except Exception as e:
                st.markdown(f"""
                <div class="error-message">
                    ❌ Error processing {uploaded_file.name}: {str(e)}
                </div>
                """, unsafe_allow_html=True)
        
        # Clean up progress indicators
        progress_bar.empty()
        status_text.empty()
        
        if new_files:
            time.sleep(0.5)  # Brief pause to show completion
            st.rerun()

def render_chat_interface():
    """Render the main chat interface"""
    
    # Display chat history
    if st.session_state.history:
        for question, answer, context, metadata in st.session_state.history:
            # User message
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>You:</strong> {question}
            </div>
            """, unsafe_allow_html=True)
            
            # Bot message
            confidence = metadata.get("confidence", 0)
            source_file = metadata.get("source_file", "Document")
            
            st.markdown(f"""
            <div class="chat-message bot-message">
                <strong>Assistant:</strong> {answer}
                <div style="margin-top: 0.5rem; font-size: 0.85em; opacity: 0.7;">
                    Confidence: {confidence:.0%} | Source: {source_file}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Input section
    st.markdown("---")
    
    # Question input
    question = st.text_input(
        "💬 Ask a question about your documents:",
        placeholder="What would you like to know?",
        key="question_input"
    )
    
    # Buttons
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        ask_button = st.button("🚀 Ask", type="primary", use_container_width=True)
    
    with col2:
        sample_button = st.button("🎲 Sample", use_container_width=True)
    
    with col3:
        if st.session_state.history:
            clear_button = st.button("🧹 Clear Chat", use_container_width=True)
        else:
            clear_button = False
    
    # Handle button clicks
    if ask_button and question:
        process_question(question)
    
    if sample_button:
        sample_questions = [
            "What is this document about?",
            "Summarize the main points",
            "What are the key conclusions?",
            "List the important details",
            "What methodology was used?"
        ]
        selected_question = sample_questions[len(st.session_state.history) % len(sample_questions)]
        process_question(selected_question)
    
    if clear_button:
        st.session_state.history = []
        st.rerun()

def process_question(question):
    """Process user question and generate response"""
    if not question.strip():
        st.warning("⚠️ Please enter a question.")
        return
    
    # Show typing indicator
    typing_placeholder = st.empty()
    typing_placeholder.markdown("""
    <div class="typing">
        🤔 AI is thinking...
    </div>
    """, unsafe_allow_html=True)
    
    try:
        start_time = time.time()
        
        # Search for relevant chunks
        search_results = st.session_state.vs.search(question, top_k=3)
        
        if not search_results["chunks"]:
            typing_placeholder.empty()
            st.error("❌ No relevant content found in your documents.")
            return
        
        # Generate answer
        context = "\n".join(search_results["chunks"])
        answer = generate_answer(context, question)
        
        end_time = time.time()
        
        # Calculate confidence based on similarity scores
        confidence = sum(search_results["scores"]) / len(search_results["scores"]) if search_results["scores"] else 0
        
        # Store metadata
        metadata = {
            "confidence": confidence,
            "source_file": search_results.get("source_files", ["Unknown"])[0],
            "response_time": end_time - start_time,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add to history
        st.session_state.history.append((question, answer, context, metadata))
        
        # Clear typing indicator and refresh
        typing_placeholder.empty()
        st.rerun()
        
    except Exception as e:
        typing_placeholder.empty()
        st.error(f"❌ Error generating answer: {str(e)}")

if __name__ == "__main__":
    main()