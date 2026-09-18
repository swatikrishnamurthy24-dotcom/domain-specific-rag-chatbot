import os
# pyrefly: ignore [missing-import]
from langchain.text_splitter import RecursiveCharacterTextSplitter
# pyrefly: ignore [missing-import]
from langchain_community.vectorstores import FAISS
# pyrefly: ignore [missing-import]
from langchain_community.embeddings import HuggingFaceEmbeddings

# Constants
INDEX_PATH = "vector_store/saved_index"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 4

# Initialize embedding model globally so it's loaded only once when the module is imported
# This caches the model efficiently.
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def create_and_save_index(documents):
    """
    Chunks the input documents and creates a FAISS vector index.
    Saves the index to the local filesystem.
    """
    if not documents:
        raise ValueError("No documents provided for indexing.")
        
    # Initialize the text splitter
    # 800 chars chunk size is a sensible default for maintaining context,
    # with 120 chars overlap to ensure sentences aren't cleanly cut off.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # Chunk the documents (metadata is automatically preserved by LangChain)
    chunks = text_splitter.split_documents(documents)
    
    # Create the FAISS vector store
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Ensure directory exists and save
    os.makedirs(INDEX_PATH, exist_ok=True)
    vector_store.save_local(INDEX_PATH)
    
    return vector_store

def load_index():
    """
    Loads the FAISS index from the local filesystem.
    Returns None if it doesn't exist.
    """
    if os.path.exists(INDEX_PATH) and os.listdir(INDEX_PATH):
        try:
            vector_store = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            return vector_store
        except Exception:
            return None
    return None

def search_index(vector_store, query: str):
    """
    Searches the given FAISS vector store for the TOP_K most similar chunks to the query.
    """
    if vector_store is None:
        raise ValueError("Vector store is not initialized.")
        
    results = vector_store.similarity_search(query, k=TOP_K)
    return results
