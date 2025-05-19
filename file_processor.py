import os
from typing import List, Generator

from langchain_community.document_loaders import ( # Updated import path
    TextLoader,
    PyPDFLoader,
    UnstructuredFileLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document # Document moved to langchain.docstore.document

import config

# Supported file types and their loaders
LOADER_MAPPING = {
    ".txt": (TextLoader, {"encoding": "utf8"}),
    ".pdf": (PyPDFLoader, {}),
    # Add more file types and loaders as needed
    # e.g., ".docx": (UnstructuredWordDocumentLoader, {}),
    # ".csv": (CSVLoader, {"encoding": "utf8"}),
}

def load_single_document(file_path: str) -> List[Document]:
    """Loads a single document from the given file path."""
    ext = "." + file_path.rsplit(".", 1)[-1].lower()
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        try:
            loader = loader_class(file_path, **loader_args)
            print(f"Loading document: {file_path}")
            return loader.load()
        except Exception as e:
            print(f"Error loading document {file_path}: {e}")
            return []
    else:
        print(f"Unsupported file type: {ext} for file {file_path}")
        # Fallback to UnstructuredFileLoader for other types if desired
        try:
            print(f"Attempting to load {file_path} with UnstructuredFileLoader.")
            loader = UnstructuredFileLoader(file_path, mode="elements")
            return loader.load()
        except Exception as e:
            print(f"Error loading document {file_path} with UnstructuredFileLoader: {e}")
            return []

def load_documents_from_path(path: str) -> List[Document]:
    """Loads documents from a given file or directory path."""
    documents = []
    if os.path.isdir(path):
        print(f"Loading documents from directory: {path}")
        for root, _, files in os.walk(path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                docs = load_single_document(file_path)
                if docs:
                    documents.extend(docs)
    elif os.path.isfile(path):
        docs = load_single_document(path)
        if docs:
            documents.extend(docs)
    else:
        print(f"Path not found: {path}")
    return documents

def split_documents(documents: List[Document]) -> List[Document]:
    """Splits a list of documents into smaller chunks."""
    print(f"Splitting {len(documents)} document(s) into chunks.")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True, # Helps in identifying the original position
    )
    chunked_documents = text_splitter.split_documents(documents)
    print(f"Successfully split documents into {len(chunked_documents)} chunks.")
    return chunked_documents

# --- Example Usage ---
def main():
    """Example of how to use the file processor."""
    # Create a dummy text file for testing
    sample_dir = "sample_docs"
    os.makedirs(sample_dir, exist_ok=True)
    sample_txt_file = os.path.join(sample_dir, "sample.txt")
    sample_pdf_file = os.path.join(sample_dir, "sample.pdf") # Assuming you have a sample.pdf

    with open(sample_txt_file, "w", encoding="utf-8") as f:
        f.write("This is the first sentence of a sample document. RAG is great.\n")
        f.write("This is the second sentence. It provides more context.\n")
        f.write("The third sentence concludes this short example for file processing.")

    print(f"--- Testing with a single .txt file: {sample_txt_file} ---")
    docs = load_documents_from_path(sample_txt_file)
    if docs:
        print(f"Loaded {len(docs)} document(s) from {sample_txt_file}.")
        for doc in docs:
            print(f"Content snippet: {doc.page_content[:100]}...")
            print(f"Metadata: {doc.metadata}")
        
        chunked_docs = split_documents(docs)
        print(f"Created {len(chunked_docs)} chunks.")
        if chunked_docs:
            print(f"First chunk content: {chunked_docs[0].page_content}")
            print(f"First chunk metadata: {chunked_docs[0].metadata}")
    else:
        print(f"No documents loaded from {sample_txt_file}.")

    print(f"\n--- Testing with a directory: {sample_dir} ---")
    # To test PDF, place a 'sample.pdf' file in the 'sample_docs' directory.
    # If you don't have one, this part might not load PDF docs.
    if not os.path.exists(sample_pdf_file):
        print(f"Note: {sample_pdf_file} not found. PDF loading test will be skipped or rely on UnstructuredFileLoader if it handles it.")
        # Create a dummy empty PDF or skip if you don't have a PDF library to create one easily for a test

    all_docs = load_documents_from_path(sample_dir)
    if all_docs:
        print(f"Loaded {len(all_docs)} document(s) from {sample_dir}.")
        chunked_all_docs = split_documents(all_docs)
        print(f"Created {len(chunked_all_docs)} chunks from all documents in {sample_dir}.")
    else:
        print(f"No documents loaded from {sample_dir}.")

    # Clean up dummy files
    # os.remove(sample_txt_file)
    # if os.path.exists(sample_pdf_file): os.remove(sample_pdf_file) # if you created one
    # os.rmdir(sample_dir)

if __name__ == "__main__":
    main()
