# Simple RAG Application with Pinecone

This application demonstrates a basic Retrieval Augmented Generation (RAG) pipeline using Pinecone as the vector database.

## Features

- Index local text files (e.g., .txt, .pdf) into a Pinecone vector index.
- Query the indexed data to answer user questions.

## Setup

1.  **Clone the repository (if applicable) or create the files as described.**

2.  **Create a Python virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    Create a `.env` file in the root directory by copying `.env.example`:
    ```bash
    copy .env.example .env  # On Windows
    # cp .env.example .env  # On macOS/Linux
    ```
    Edit the `.env` file and add your API keys:
    ```
    PINECONE_API_KEY="YOUR_PINECONE_API_KEY"
    PINECONE_ENVIRONMENT="YOUR_PINECONE_ENVIRONMENT" # e.g., "gcp-starter" or your specific environment
    OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
    PINECONE_INDEX_NAME="your-rag-index" # Choose a name for your Pinecone index
    ```

5.  **Ensure you have a Pinecone account and an OpenAI account for API keys.**

## Usage

(Instructions will be updated as the application is built.)

To run the application:
```bash
python app.py
```

You will be prompted to either index files or ask a question.

### Indexing Files

-   Choose the 'index' option.
-   Provide the path to a file or a directory containing files to be indexed.

### Asking Questions

-   Choose the 'query' option.
-   Enter your question.
