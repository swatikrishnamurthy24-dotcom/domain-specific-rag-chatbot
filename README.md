# Domain-Specific RAG Chatbot

## Project Objective
A reliable Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDF documents and ask questions about their contents. The chatbot answers questions **strictly** using information retrieved from the uploaded PDFs and prevents hallucination.

## Problem Statement
Traditional LLMs often hallucinate or provide generalized answers when queried about domain-specific knowledge. This project addresses this by ensuring that the chatbot relies solely on the provided documentation to formulate its answers, providing accurate source citations and admitting when it does not know the answer.

## Features
- **PDF Upload:** Upload one or multiple PDF documents.
- **Robust Text Extraction:** Extracts text page-by-page while preserving document name and page number metadata.
- **Smart Chunking:** Splits text into optimal chunks (800 characters) for effective retrieval.
- **Vector Search:** Uses Sentence Transformers (`all-MiniLM-L6-v2`) and FAISS for fast similarity search.
- **Strict Guardrails:** Prevents hallucinations by strictly grounding the LLM (Groq) on retrieved context.
- **Source Citations:** Accurately displays the document name and page number for the information used in the answer.

## Architecture
1. **UI:** Streamlit
2. **Document Loader:** `pypdf`
3. **Text Chunking:** LangChain RecursiveCharacterTextSplitter
4. **Embedding Model:** Sentence Transformers
5. **Vector Store:** FAISS
6. **Retriever:** LangChain FAISS Retriever
7. **LLM:** Groq API

## Folder Structure
```text
domain_rag_chatbot/
│
├── app.py                  # Main Streamlit application
├── document_loader.py      # PDF parsing and metadata preservation
├── vector_store.py         # Chunking, Embeddings, FAISS index management
├── prompt.py               # Strict guardrail prompts
├── rag_pipeline.py         # LLM integration and context assembly
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── .env.example            # Environment variables template
├── .gitignore              # Git ignore file
│
├── documents/              # Directory for sample uploaded PDFs
├── vector_store/
│   └── saved_index/        # Directory for the persistent FAISS index
└── tests/
    └── test_questions.csv  # 15 Test cases
```

## Installation Steps

1. **Clone or Download the Repository:**
   Navigate to the project folder.

2. **Create a Virtual Environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the Virtual Environment:**
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Environment Variable Setup:**
   - Copy `.env.example` to a new file named `.env`.
   - Open `.env` and add your Groq API key:
     ```
     GROQ_API_KEY=your_actual_api_key_here
     ```

## How to Run the Application
Start the Streamlit server:
```bash
streamlit run app.py
```

## How Document Processing Works
- The user uploads PDFs via the Streamlit sidebar.
- `document_loader.py` uses `pypdf` to extract text from each page. It attaches the source filename and the page number to the extracted text block.
- These chunks are then fed into LangChain's `RecursiveCharacterTextSplitter`.

## How Retrieval Works
- The extracted text is converted into vector embeddings using `all-MiniLM-L6-v2`.
- These vectors are stored in a FAISS index (`vector_store.py`).
- When a user asks a question, the question is embedded, and a similarity search is performed against the FAISS index to retrieve the top `k` most relevant chunks.

## How Hallucination is Prevented
- The `prompt.py` module defines a strict system prompt instructing the LLM to only answer based on the provided context.
- If the required information is not present, the LLM is instructed to respond: "I could not find this information in the uploaded documents."

## Testing Instructions
- Review `tests/test_questions.csv` for a suite of 15 questions covering various edge cases.
- Upload a test PDF, ask the questions, and verify that the application properly cites sources and refuses to answer unrelated questions.

## Security Considerations
- **API Keys:** Never hardcoded; always loaded via `.env`.
- **Data Privacy:** PDFs are processed locally. Only retrieved context is sent to the LLM.
- **Prompt Injection:** The system prompt instructs the LLM to treat document content as data, ignoring instructions embedded within the PDF.

## Future Enhancements
- Support for other document types (DOCX, TXT).
- Implementation of OCR for scanned PDFs.
- Hybrid search (combining keyword and vector search).

## Viva Questions
1. **What is RAG?** Retrieval-Augmented Generation. It enhances an LLM's responses by fetching relevant information from a custom knowledge base before generating the answer.
2. **Why FAISS?** FAISS (Facebook AI Similarity Search) is highly optimized for fast dense vector similarity search, which is crucial for low-latency retrieval.
3. **How do you prevent hallucination?** By providing a strict system prompt that limits the LLM to only the provided context and instructs it to admit when it lacks information.
4. **Why use LangChain?** LangChain provides excellent abstractions for chunking text, managing prompt templates, and orchestrating the RAG pipeline.
