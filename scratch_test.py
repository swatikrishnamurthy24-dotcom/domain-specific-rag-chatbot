import os
from io import BytesIO
from pypdf import PdfWriter, PdfReader
from langchain.schema import Document
from document_loader import extract_text_from_pdfs
from vector_store import create_and_save_index, load_index, search_index
from rag_pipeline import generate_answer
from dotenv import load_dotenv

load_dotenv()

# Create a sample PDF in memory
writer = PdfWriter()
writer.add_blank_page(width=200, height=200) # This won't have text easily unless we use ReportLab, so let's mock the document_loader logic directly for the text chunking part

# But wait, pypdf extraction needs real text. 
# Let's bypass pypdf for a moment and just feed Documents to the vector_store to test the FAISS + Groq pipeline.

docs = [
    Document(page_content="Employees receive 12 casual leaves per year. The CEO of the company is Alice Smith.", metadata={"source": "Company_Policy.pdf", "page": 6}),
    Document(page_content="Training procedure requires 3 weeks of onboarding.", metadata={"source": "Company_Policy.pdf", "page": 3})
]

print("1. Creating Vector Store Index...")
vector_store = create_and_save_index(docs)
print("Index created and saved.")

print("2. Loading Vector Store Index...")
loaded_store = load_index()
if loaded_store is None:
    print("Failed to load index!")
    exit(1)
print("Index loaded successfully.")

print("3. Testing Retrieval and Answer Generation...")
query1 = "How many casual leaves do employees get?"
print(f"Query: {query1}")
chunks = search_index(loaded_store, query1)
answer, sources = generate_answer(query1, chunks)
print(f"Answer: {answer}")
print(f"Sources: {sources}")

print("-" * 40)

query2 = "What is the training procedure?"
print(f"Query: {query2}")
chunks = search_index(loaded_store, query2)
answer, sources = generate_answer(query2, chunks)
print(f"Answer: {answer}")
print(f"Sources: {sources}")

print("-" * 40)

query3 = "What is the company's revenue?"
print(f"Query: {query3}")
chunks = search_index(loaded_store, query3)
answer, sources = generate_answer(query3, chunks)
print(f"Answer: {answer}")
print(f"Sources: {sources}")

print("Tests completed successfully.")
