import os
import sys
from dotenv import load_dotenv

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import config
import rag_core
from vector_store_manager import delete_pinecone_index, describe_index_stats, get_pinecone_index

# --- Test Configuration ---
# Ensure this PDF file exists at the specified path relative to this script
PDF_FILE_PATH = os.path.join(project_root, "Semantic-categorization-eligibility-criteria.pdf")
TEST_QNA_INDEX_NAME = "cascade-qna-test-index" # A dedicated index for this Q&A test

# Define a test question relevant to your PDF's content
# You can change this question to better suit your PDF.
TEST_QUESTION = "What are the eligibility criteria for semantic categorization?"

# Set to True to delete the TEST_QNA_INDEX_NAME after the test, False to keep it.
CLEAN_UP_INDEX_AFTER_TEST = True
# Set to True to delete the index if it exists before running the test for a clean slate.
CLEAN_UP_BEFORE_TEST = True 

def run_qna_test():
    """Runs the Q&A test: indexes a PDF and asks a question."""
    print("--- Starting Q&A Test ---")
    print(f"INFO: This test will use a dedicated Pinecone index: '{TEST_QNA_INDEX_NAME}'")
    if CLEAN_UP_INDEX_AFTER_TEST:
        print(f"INFO: Test index ('{TEST_QNA_INDEX_NAME}') will be deleted after the test.")
    if CLEAN_UP_BEFORE_TEST:
        print(f"INFO: Test index ('{TEST_QNA_INDEX_NAME}') will be deleted BEFORE the test if it exists.")

    original_pinecone_index_name = getattr(config, 'PINECONE_INDEX_NAME', None)
    config.PINECONE_INDEX_NAME = TEST_QNA_INDEX_NAME # Override for this test

    try:
        # 1. Check Core Configuration (API Keys, Environment)
        if not config.PINECONE_API_KEY or not config.PINECONE_ENVIRONMENT or not config.OPENAI_API_KEY:
            raise ValueError("PINECONE_API_KEY, PINECONE_ENVIRONMENT, or OPENAI_API_KEY not found in .env or config.")
        print("Core API Key configuration check passed.")
        print(f"Using temporary test Pinecone Index: {config.PINECONE_INDEX_NAME}")

        # 2. Check if PDF file exists
        if not os.path.exists(PDF_FILE_PATH):
            print(f"ERROR: PDF file not found at '{PDF_FILE_PATH}'.")
            return
        print(f"PDF file for Q&A test: {PDF_FILE_PATH}")

        # 3. Optional: Clean up existing test index before starting
        if CLEAN_UP_BEFORE_TEST:
            print(f"\n--- Pre-test Cleanup: Checking for existing index '{TEST_QNA_INDEX_NAME}' ---")
            try:
                # Check if index exists by trying to describe it
                # get_pinecone_index will also list indexes, let's use delete_pinecone_index directly
                # which has its own existence check.
                print(f"Attempting to delete '{TEST_QNA_INDEX_NAME}' if it exists to ensure a clean run...")
                deleted_before = delete_pinecone_index(TEST_QNA_INDEX_NAME, suppress_confirm=True) # Suppress confirmation for automated cleanup
                if deleted_before:
                    print(f"Pre-existing test index '{TEST_QNA_INDEX_NAME}' deleted.")
                else:
                    print(f"Test index '{TEST_QNA_INDEX_NAME}' not found or not deleted before test.") 
            except Exception as e:
                print(f"Error during pre-test cleanup of index '{TEST_QNA_INDEX_NAME}': {e}")

        # 4. Run the indexing process for the Q&A test
        print("\n--- Indexing PDF for Q&A Test ---")
        indexing_success = rag_core.index_data_from_path(PDF_FILE_PATH)

        if not indexing_success:
            print("\nFAILURE: PDF indexing process failed. Cannot proceed with Q&A.")
            return
        print("\nSUCCESS: PDF indexing process completed for Q&A test.")

        # 5. Ask the test question
        print("\n--- Asking Test Question ---")
        print(f"Question: {TEST_QUESTION}")
        answer = rag_core.answer_query(TEST_QUESTION) # Corrected function name
        
        print("\n--- Answer Received ---")
        if answer:
            print(f"Answer: {answer}")
        else:
            print("No answer was returned from the RAG system.")

    except ValueError as ve:
        print(f"CRITICAL CONFIGURATION ERROR: {ve}")
    except Exception as e:
        print(f"UNEXPECTED ERROR during Q&A Test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 6. Optional: Clean up (delete the test index) after test
        if CLEAN_UP_INDEX_AFTER_TEST:
            print("\n--- Post-test Cleanup: Deleting Test Index ---")
            try:
                print(f"Attempting to delete test index '{config.PINECONE_INDEX_NAME}'...")
                deleted_after = delete_pinecone_index(config.PINECONE_INDEX_NAME, suppress_confirm=True)
                if deleted_after:
                    print(f"Test index '{config.PINECONE_INDEX_NAME}' deleted successfully.")
                else:
                    print(f"Failed to delete test index '{config.PINECONE_INDEX_NAME}'.")
            except Exception as e:
                print(f"Error during post-test cleanup of index '{config.PINECONE_INDEX_NAME}': {e}")
        else:
            print(f"\nSkipping post-test index cleanup. Index '{config.PINECONE_INDEX_NAME}' was not deleted.")
        
        # Restore original PINECONE_INDEX_NAME from config
        if original_pinecone_index_name is not None:
            config.PINECONE_INDEX_NAME = original_pinecone_index_name
        elif hasattr(config, 'PINECONE_INDEX_NAME'):
            delattr(config, 'PINECONE_INDEX_NAME')
        print(f"Restored original PINECONE_INDEX_NAME config. Current is: {getattr(config, 'PINECONE_INDEX_NAME', 'Not Set')}")
        print("\n--- Q&A Test Finished ---")

if __name__ == "__main__":
    # Ensure .env is loaded
    env_path = os.path.join(project_root, '.env')
    if os.path.exists(env_path):
        load_dotenv(dotenv_path=env_path)
        print(f"test_qna.py: .env file loaded successfully from {env_path}.")
    else:
        print(f"test_qna.py: .env file not found at {env_path}. Relying on environment variables if set, or defaults.")
    
    # Pinecone client, Embeddings, and LLM are initialized automatically when 'config.py'
    # (or modules like rag_core.py that import config) are imported.
    # The success messages for these initializations are printed from those modules.
    # Therefore, no explicit re-initialization is needed here.

    run_qna_test()
