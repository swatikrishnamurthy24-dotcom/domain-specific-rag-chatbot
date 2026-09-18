import os
from io import BytesIO
from pypdf import PdfReader
from langchain.schema import Document

def extract_text_from_pdfs(uploaded_files) -> list[Document]:
    """
    Extracts text from a list of uploaded Streamlit PDF files.
    Preserves document name and page number metadata for every page.
    Skips empty pages.
    """
    documents = []
    
    for uploaded_file in uploaded_files:
        # Streamlit UploadedFile has a .name attribute
        file_name = uploaded_file.name
        
        # Read the file directly from memory
        pdf_bytes = uploaded_file.read()
        pdf_stream = BytesIO(pdf_bytes)
        
        try:
            pdf_reader = PdfReader(pdf_stream)
            
            for page_num, page in enumerate(pdf_reader.pages, start=1):
                text = page.extract_text()
                
                # Clean up extracted text a bit
                if text:
                    text = text.strip()
                
                # Only add non-empty pages
                if text:
                    # Create a LangChain Document with metadata
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": file_name,
                            "page": page_num
                        }
                    )
                    documents.append(doc)
        except Exception as e:
            # Re-raise with filename for better error handling upstream
            raise RuntimeError(f"Error processing {file_name}: {str(e)}")
            
    return documents
