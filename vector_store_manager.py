from pinecone import Pinecone, ServerlessSpec, PodSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings # Or your preferred embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document # Document moved to langchain.docstore.document
from typing import List

import config

# Initialize Pinecone client
pc = None
if config.PINECONE_API_KEY and config.PINECONE_ENVIRONMENT:
    try:
        pc = Pinecone(api_key=config.PINECONE_API_KEY) # environment is deprecated, use cloud and region for serverless or just api_key for pod-based
        print("Pinecone client initialized successfully.")
    except Exception as e:
        print(f"Error initializing Pinecone client: {e}")
        pc = None
else:
    print("Pinecone API Key or Environment not found in config. Skipping Pinecone client initialization.")

# Initialize Embeddings
# You can choose between OpenAI and HuggingFace embeddings here
if config.OPENAI_API_KEY:
    embeddings_model = OpenAIEmbeddings(
        model="text-embedding-ada-002", # A common OpenAI embedding model
        openai_api_key=config.OPENAI_API_KEY
    )
    print(f"Using OpenAIEmbeddings model: text-embedding-ada-002")
else:
    # Fallback or default to HuggingFace embeddings if OpenAI key is not available
    # Ensure sentence-transformers is installed: pip install sentence-transformers
    embeddings_model = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
    print(f"OpenAI API key not found. Using HuggingFaceEmbeddings model: {config.EMBEDDING_MODEL_NAME}")

def get_pinecone_index(index_name: str, recreate_if_exists: bool = False):
    """Gets a Pinecone index object. Creates it if it doesn't exist."""
    if not pc:
        raise ConnectionError("Pinecone client not initialized. Check API key and environment.")

    # Get list of existing index names compatible with pinecone-client v3.x and v4.x+
    existing_indexes_list = pc.list_indexes()
    current_index_names = []
    if hasattr(existing_indexes_list, 'names'): # Handles v3.x style
        if callable(existing_indexes_list.names):
            current_index_names = existing_indexes_list.names() # Call the method for v3.x
        else:
            current_index_names = existing_indexes_list.names # Fallback if it's an attribute (less likely for .names in v3.x)
    elif hasattr(existing_indexes_list, 'indexes') and existing_indexes_list.indexes is not None: # Handles v4.x+ style (List[IndexModel])
        # In pinecone-client 4.x+, existing_indexes_list.indexes should be List[IndexModel]
        # Each IndexModel object has a .name attribute.
        idx_list = existing_indexes_list.indexes
        if isinstance(idx_list, list):
            current_index_names = [idx_model.name for idx_model in idx_list if hasattr(idx_model, 'name')]
        else:
            print(f"Warning: existing_indexes_list.indexes was not a list (type: {type(idx_list)}). Unable to get index names.")
    else: # Fallback for other possible structures or if it's just a list of Index objects
        try:
            current_index_names = [idx.name for idx in existing_indexes_list]
        except Exception:
            print("Could not determine current index names from pc.list_indexes() response. Proceeding with caution.")

    if recreate_if_exists and index_name in current_index_names:
        print(f"Recreating index: Deleting existing index '{index_name}'...")
        delete_pinecone_index(index_name, suppress_confirm=True)
        # After deletion, the index is no longer in current_index_names for the 'if' below,
        # so it will proceed to creation. We can also optimistically remove it from the list:
        if index_name in current_index_names: # Re-check in case delete failed silently
             current_index_names.remove(index_name)

    if index_name not in current_index_names:
        print(f"Index '{index_name}' not found. Creating a new one...")
        # Determine embedding dimension based on the model used
        if isinstance(embeddings_model, OpenAIEmbeddings):
            # Example: text-embedding-ada-002 has 1536 dimensions
            # It's better to get this programmatically if possible or ensure it's known
            dimension = 1536 
        elif isinstance(embeddings_model, HuggingFaceEmbeddings):
            # For HuggingFace models, the dimension depends on the specific model
            # e.g., all-MiniLM-L6-v2 has 384 dimensions
            # This is a common way to get the dimension, but might need adjustment
            try:
                sample_embedding = embeddings_model.embed_query("sample text")
                dimension = len(sample_embedding)
            except Exception as e:
                print(f"Could not determine embedding dimension for {config.EMBEDDING_MODEL_NAME}: {e}")
                # Fallback to a common dimension or raise error
                dimension = 384 # Default for all-MiniLM-L6-v2
                print(f"Assuming dimension {dimension} for {config.EMBEDDING_MODEL_NAME}.")
        else:
            raise ValueError("Unsupported embedding model type to determine dimension.")

        try:
            # Example for creating a serverless index (newer Pinecone offering)
            # Adjust cloud and region as per your Pinecone setup
            # For pod-based indexes, the creation logic is different.
            # Check Pinecone documentation for the latest way to create indexes.
            # pc.create_index(
            #     name=index_name,
            #     dimension=dimension,
            #     metric="cosine", # Common metric for semantic similarity
            #     spec=ServerlessSpec(
            #         cloud='aws', # or 'gcp', 'azure'
            #         region='us-west-2' # your pinecone region
            #     )
            # )
            # Determine cloud provider (this is an assumption, might need refinement)
            cloud_provider = 'aws' # Defaulting to AWS for regions like 'us-east-1'
            if 'gcp' in config.PINECONE_ENVIRONMENT.lower() or 'google' in config.PINECONE_ENVIRONMENT.lower():
                cloud_provider = 'gcp'
            elif 'azure' in config.PINECONE_ENVIRONMENT.lower():
                cloud_provider = 'azure'
            
            print(f"Creating Serverless index '{index_name}' in cloud '{cloud_provider}' region '{config.PINECONE_ENVIRONMENT}' with dimension {dimension}.")
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=cloud_provider,
                    region=config.PINECONE_ENVIRONMENT 
                )
            )
            print(f"Index '{index_name}' created successfully with dimension {dimension}.")
        except Exception as e:
            print(f"Error creating index '{index_name}': {e}")
            raise
    else:
        print(f"Index '{index_name}' already exists.")
    
    return pc.Index(index_name)

def get_vector_store(index_name: str = config.PINECONE_INDEX_NAME, recreate_if_exists: bool = False) -> PineconeVectorStore:
    """Initializes and returns a PineconeVectorStore instance."""
    if not pc:
        raise ConnectionError("Pinecone client not initialized.")
    if not embeddings_model:
        raise ValueError("Embeddings model not initialized.")

    # Ensure the index exists (or create/recreate it as per recreate_if_exists)
    get_pinecone_index(index_name, recreate_if_exists=recreate_if_exists)
    
    try:
        vector_store = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings_model,
            # namespace= # Optional: if you want to use namespaces within your index
        )
        print(f"Successfully connected to Pinecone index '{index_name}' as a vector store.")
        return vector_store
    except Exception as e:
        print(f"Error connecting to existing Pinecone index '{index_name}': {e}")
        # Potentially try to initialize if 'from_existing_index' fails for a new index
        # though get_pinecone_index should handle creation.
        # For now, we'll re-raise as the expectation is the index should be ready.
        raise

def upsert_documents(index_name: str, documents: List[Document], batch_size: int = 100):
    """Upserts documents into the specified Pinecone index."""
    if not documents:
        print("No documents to upsert.")
        return

    vector_store = get_vector_store(index_name)
    print(f"Upserting {len(documents)} documents to index '{index_name}'...")
    
    # Langchain's PineconeVectorStore.add_documents handles batching internally
    # but it's good to be mindful of Pinecone's vector limits per request if doing manually.
    try:
        vector_store.add_documents(documents, batch_size=batch_size)
        print(f"Successfully upserted {len(documents)} documents.")
    except Exception as e:
        print(f"Error upserting documents: {e}")
        raise

def query_vector_store(index_name: str, query_text: str, k: int = 5) -> List[Document]:
    """Queries the vector store for similar documents."""
    vector_store = get_vector_store(index_name)
    print(f"Querying index '{index_name}' with: '{query_text[:50]}...' (k={k})")
    try:
        results = vector_store.similarity_search(query_text, k=k)
        print(f"Found {len(results)} similar documents.")
        return results
    except Exception as e:
        print(f"Error querying vector store: {e}")
        raise

def describe_index_stats(index_name: str):
    """Prints statistics about the Pinecone index."""
    if not pc:
        print("Pinecone client not initialized.")
        return
    try:
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        print(f"Index '{index_name}' stats: {stats}")
        return stats
    except Exception as e:
        print(f"Error describing index stats for '{index_name}': {e}")
        return None

def delete_pinecone_index(index_name: str, suppress_confirm: bool = False):
    """Deletes the specified Pinecone index."""
    if not pc:
        print("Pinecone client not initialized.")
        return False
    existing_indexes_list = pc.list_indexes()
    current_index_names = []
    if hasattr(existing_indexes_list, 'names'): # Handles v3.x style
        if callable(existing_indexes_list.names):
            current_index_names = existing_indexes_list.names() # Call the method for v3.x
        else:
            current_index_names = existing_indexes_list.names # Fallback if it's an attribute (less likely for .names in v3.x)
    elif hasattr(existing_indexes_list, 'indexes') and existing_indexes_list.indexes is not None: # Handles v4.x+ style (List[IndexModel])
        # In pinecone-client 4.x+, existing_indexes_list.indexes should be List[IndexModel]
        # Each IndexModel object has a .name attribute.
        idx_list = existing_indexes_list.indexes
        if isinstance(idx_list, list):
            current_index_names = [idx_model.name for idx_model in idx_list if hasattr(idx_model, 'name')]
        else:
            print(f"Warning: existing_indexes_list.indexes was not a list (type: {type(idx_list)}). Unable to get index names.")
    else: # Fallback for other possible structures or if it's just a list of Index objects
        try:
            current_index_names = [idx.name for idx in existing_indexes_list]
        except Exception:
            print("Could not determine current index names from pc.list_indexes() response for deletion. Proceeding with caution.")

    if index_name in current_index_names:
        try:
            print(f"Deleting index '{index_name}'...")
            pc.delete_index(index_name)
            print(f"Index '{index_name}' deleted successfully.")
            return True
        except Exception as e:
            print(f"Error deleting index '{index_name}': {e}")
            return False
    else:
        print(f"Index '{index_name}' not found, cannot delete.")
        return False

# --- Example Usage ---
def main():
    """Example usage of the vector store manager."""
    # Ensure config.py has PINECONE_API_KEY, PINECONE_ENVIRONMENT, and OPENAI_API_KEY (optional)
    # and PINECONE_INDEX_NAME set in your .env file.
    try:
        config.check_config() # Basic check from config.py
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Please ensure your .env file is set up correctly.")
        return

    if not pc:
        print("Pinecone client could not be initialized. Exiting example.")
        return

    test_index_name = config.PINECONE_INDEX_NAME
    print(f"\n--- Testing with Pinecone Index: {test_index_name} ---")

    try:
        # 1. Get/Create Index & Vector Store
        vector_store = get_vector_store(test_index_name)
        print(f"Vector store obtained for index: {test_index_name}")

        # 2. Describe Index Stats (initially likely empty or non-existent)
        describe_index_stats(test_index_name)

        # 3. Upsert Sample Documents
        print("\n--- Upserting sample documents ---")
        sample_docs = [
            Document(page_content="The quick brown fox jumps over the lazy dog.", metadata={"source": "test-doc", "id": "1"}),
            Document(page_content="Pinecone is a vector database for fast similarity search.", metadata={"source": "test-doc", "id": "2"}),
            Document(page_content="Retrieval Augmented Generation enhances LLM responses.", metadata={"source": "test-doc", "id": "3"})
        ]
        upsert_documents(test_index_name, sample_docs)
        
        # Describe stats again to see the change
        describe_index_stats(test_index_name)

        # 4. Query Documents
        print("\n--- Querying documents ---")
        query = "What is RAG?"
        results = query_vector_store(test_index_name, query, k=2)
        if results:
            for i, doc in enumerate(results):
                print(f"Result {i+1}: {doc.page_content} (Metadata: {doc.metadata})")
        else:
            print("No results found for the query.")
        
        query2 = "Tell me about a canine."
        results2 = query_vector_store(test_index_name, query2, k=1)
        if results2:
            print(f"Result for '{query2}': {results2[0].page_content}")

    except ConnectionError as ce:
        print(f"Connection Error: {ce}")
    except ValueError as ve:
        print(f"Value Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred during the example: {e}")
    finally:
        # 5. Optionally delete the index (useful for testing)
        confirm_delete = 'yes' # Default to yes if confirmation is suppressed
        if not suppress_confirm:
            confirm_delete = input(f"Are you sure you want to delete the index '{test_index_name}'? This action cannot be undone. (yes/no): ").lower()
        if confirm_delete == 'yes':
            delete_pinecone_index(test_index_name, suppress_confirm=True)
        else:
            print(f"Index '{test_index_name}' was not deleted.")
        print(f"Example finished. If you want to clean up, run `python vector_store_manager.py` and choose to delete, or manually delete from Pinecone console.")

if __name__ == "__main__":
    main()
