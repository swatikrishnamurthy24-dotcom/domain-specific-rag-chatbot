import os
import time
from dotenv import load_dotenv
from google import genai

load_dotenv()
from prompt import prompt_template

def format_context(docs):
    """
    Formats the retrieved documents into a single string for the context.
    """
    return "\n\n".join([doc.page_content for doc in docs])

def get_unique_sources(docs):
    """
    Extracts unique sources (document name and page) from the retrieved documents.
    Prevents duplicate citations if multiple chunks come from the same page.
    """
    sources = set()
    for doc in docs:
        source = doc.metadata.get('source', 'Unknown Document')
        page = doc.metadata.get('page', 'Unknown Page')
        sources.add(f"• {source} — Page {page}")
    return sorted(list(sources))

def generate_answer(query: str, retrieved_docs: list) -> tuple[str, list[str]]:
    """
    Takes the user query and the retrieved documents, formats them into the prompt,
    and calls the Gemini LLM to generate the final grounded answer.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is missing. Please set it in the .env file.")
        
    if not retrieved_docs:
        return "I could not find this information in the uploaded documents.", []
        
    context = format_context(retrieved_docs)
    unique_sources = get_unique_sources(retrieved_docs)
    
    # Initialize the Gemini LLM
    # Use environment variable for model configuration
    model_name = os.getenv("GEMINI_MODEL")
    if not model_name:
        raise ValueError("GEMINI_MODEL environment variable is missing. Please set it in the .env file.")
        
    print(f"Using Gemini model: {model_name}")
    
    # Format the prompt
    final_prompt = prompt_template.format(context=context, question=query)
    
    # Generate the response
    max_retries = 3
    base_delay = 2
    for attempt in range(max_retries):
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=model_name,
                contents=final_prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0.0
                )
            )
            return response.text, unique_sources
        except Exception as e:
            error_str = str(e)
            if "503" in error_str and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"503 UNAVAILABLE encountered. Retrying in {delay} seconds...")
                time.sleep(delay)
                continue
            raise RuntimeError(f"Error communicating with the LLM API: {error_str}")
