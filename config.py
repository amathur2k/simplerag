import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Pinecone Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT") # Pinecone renamed 'environment' to 'cloud' and 'region' for serverless indexes
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Embedding Model Configuration (example using Hugging Face Sentence Transformers)
# You can switch to OpenAI embeddings if preferred, adjust rag_core.py accordingly
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# LLM Configuration (example using OpenAI)
LLM_MODEL_NAME = "gpt-4o-mini"

# Document Processing Configuration
CHUNK_SIZE = 1000  # Size of text chunks in characters
CHUNK_OVERLAP = 100 # Number of characters to overlap between chunks

# --- Sanity Checks --- 
def check_config():
    """Checks if essential configurations are set."""
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is not set. Please check your .env file.")
    if not PINECONE_ENVIRONMENT:
        raise ValueError("PINECONE_ENVIRONMENT is not set. Please check your .env file.")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set. Please check your .env file.")
    if not PINECONE_INDEX_NAME:
        raise ValueError("PINECONE_INDEX_NAME is not set in .env file. Please add it, e.g., PINECONE_INDEX_NAME='your-actual-index-name'.")
    print("Configuration check passed.")

if __name__ == "__main__":
    # This will run if you execute config.py directly, e.g., python config.py
    try:
        check_config()
        print(f"Pinecone Index Name: {PINECONE_INDEX_NAME}")
        print(f"Embedding Model: {EMBEDDING_MODEL_NAME}")
        print(f"LLM Model: {LLM_MODEL_NAME}")
    except ValueError as e:
        print(f"Configuration Error: {e}")
