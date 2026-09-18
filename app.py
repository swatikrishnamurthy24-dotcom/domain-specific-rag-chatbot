import os
import streamlit as st
from dotenv import load_dotenv
from document_loader import extract_text_from_pdfs
from vector_store import create_and_save_index, load_index, search_index
from rag_pipeline import generate_answer

# Load environment variables (like GEMINI_API_KEY)
load_dotenv()

# --- Page Config & UI Setup ---
st.set_page_config(
    page_title="Domain-Specific RAG Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom minimal CSS for source styling
st.markdown("""
<style>
.source-box {
    background-color: #f0f2f6;
    padding: 10px;
    border-radius: 5px;
    font-size: 0.85em;
    color: #333;
    margin-top: 5px;
    border-left: 3px solid #0052cc;
}
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "index_loaded" not in st.session_state:
    st.session_state.index_loaded = False
if "processing" not in st.session_state:
    st.session_state.processing = False

# Try to load existing index on startup
if not st.session_state.index_loaded:
    existing_index = load_index()
    if existing_index is not None:
        st.session_state.index_loaded = True

# --- Sidebar: Document Upload & Processing ---
with st.sidebar:
    st.header("📄 Document Management")
    uploaded_files = st.file_uploader(
        "Upload PDF documents", 
        type="pdf", 
        accept_multiple_files=True,
        help="Upload one or multiple PDF files to serve as the knowledge base."
    )
    
    if uploaded_files:
        st.write(f"**{len(uploaded_files)} file(s) selected.**")
        for f in uploaded_files:
            st.text(f"- {f.name}")
            
    if st.button("Process Documents", type="primary", disabled=st.session_state.processing):
        if not uploaded_files:
            st.error("Please upload at least one PDF.")
        else:
            try:
                st.session_state.processing = True
                with st.spinner("Extracting text from PDFs..."):
                    docs = extract_text_from_pdfs(uploaded_files)
                    
                if not docs:
                    st.error("No extractable text found in the uploaded PDFs.")
                else:
                    with st.spinner("Generating embeddings and creating FAISS index..."):
                        create_and_save_index(docs)
                        
                    st.session_state.index_loaded = True
                    st.success(f"Successfully processed {len(docs)} chunks from {len(uploaded_files)} document(s)!")
            except Exception as e:
                st.error(f"Error during processing: {str(e)}")
            finally:
                st.session_state.processing = False

    st.markdown("---")
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

    if st.session_state.index_loaded:
        st.success("✅ Knowledge base is ready.")
    else:
        st.warning("⚠️ No documents processed yet.")

# --- Main Chat Area ---
st.title("📚 Domain-Specific RAG Chatbot")
st.subheader("Ask questions strictly based on your uploaded PDF documents")

st.markdown("""
> **Note:** You are a document question-answering assistant.Answer only from the supplied context. If the answer is not available,say:"I could not find this information in the
uploaded documents." Do not invent facts.Mention the source document and page number when available.
""")

# Display Chat History
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])
        if chat["role"] == "assistant" and chat.get("sources"):
            sources_html = "<br>".join(chat["sources"])
            st.markdown(f'<div class="source-box"><strong>Sources:</strong><br>{sources_html}</div>', unsafe_allow_html=True)

# Chat Input
query = st.chat_input("Ask a question about your documents...")

if query:
    # Append user question
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
        
    # Check prerequisites
    if not os.getenv("GEMINI_API_KEY"):
        with st.chat_message("assistant"):
            st.error("GEMINI_API_KEY is missing. Please set it in your .env file.")
    elif not st.session_state.index_loaded:
        with st.chat_message("assistant"):
            st.warning("Please upload and process documents before asking questions.")
    else:
        # Process Question
        with st.chat_message("assistant"):
            with st.spinner("Searching for relevant information..."):
                try:
                    # 1. Load the index
                    vector_store = load_index()
                    if vector_store is None:
                        raise ValueError("Failed to load vector store. Please reprocess the documents.")
                        
                    # 2. Retrieve chunks
                    retrieved_chunks = search_index(vector_store, query)
                    
                    # 3. Generate Answer
                    answer, sources = generate_answer(query, retrieved_chunks)
                    
                    # 4. Display Answer and Sources
                    st.markdown(answer)
                    if sources:
                        sources_html = "<br>".join(sources)
                        st.markdown(f'<div class="source-box"><strong>Sources:</strong><br>{sources_html}</div>', unsafe_allow_html=True)
                        
                    # 5. Append to history
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": sources
                    })
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
