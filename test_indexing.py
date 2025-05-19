import os
import sys
from dotenv import load_dotenv

# Attempt to load .env file explicitly for this script
if load_dotenv():
    print("test_indexing.py: .env file loaded successfully.")
else:
    print("test_indexing.py: .env file not found or failed to load. Ensure it's in the project root.")

# Ensure the main project directory is in the Python path
# This allows a_test_python_version_results to find modules like config, rag_core, etc.
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import rag_core
import config # For checking config and accessing PINECONE_INDEX_NAME
from vector_store_manager import describe_index_stats, delete_pinecone_index

# --- Configuration for the test ---
# Use an absolute path for the PDF file to avoid ambiguity
PDF_FILE_PATH = r"C:\Users\ADMIN\OneDrive\learn\simplerag3\Semantic-categorization-eligibility-criteria.pdf"
# To make it more portable, you could also construct it like:
# PDF_FILE_PATH = os.path.join(project_root, "Semantic-categorization-eligibility-criteria.pdf")
# This assumes the PDF is in the same directory as this test script or the main project dir.

# --- Test-specific Pinecone Index Configuration ---
TEST_SPECIFIC_INDEX_NAME = "cascade-temp-test-index" # Hardcoded for this test
CLEAN_UP_INDEX_AFTER_TEST = True # Default to True for a temporary test index
                                  # Be cautious if you change TEST_SPECIFIC_INDEX_NAME to something important.

def run_indexing_test():
    """Runs the indexing test for the specified PDF file."""
    print("--- Starting PDF Indexing Test ---")
    print(f"INFO: This test will use a dedicated Pinecone index: '{TEST_SPECIFIC_INDEX_NAME}'")
    if CLEAN_UP_INDEX_AFTER_TEST:
        print(f"INFO: This index ('{TEST_SPECIFIC_INDEX_NAME}') will be deleted after the test.")

    original_pinecone_index_name = getattr(config, 'PINECONE_INDEX_NAME', None)
    config.PINECONE_INDEX_NAME = TEST_SPECIFIC_INDEX_NAME # Override for this test

    try:
        # 1. Check Core Configuration (API Keys, Environment)
        # The .env still needs to be valid for PINECONE_API_KEY, PINECONE_ENVIRONMENT, OPENAI_API_KEY
        if not config.PINECONE_API_KEY or not config.PINECONE_ENVIRONMENT or not config.OPENAI_API_KEY:
            raise ValueError("PINECONE_API_KEY, PINECONE_ENVIRONMENT, or OPENAI_API_KEY not found in .env or config.")
        print("Core API Key configuration check passed (PINECONE_API_KEY, PINECONE_ENVIRONMENT, OPENAI_API_KEY).")
        print(f"Using temporary test Pinecone Index: {config.PINECONE_INDEX_NAME}")

        # 2. Check if PDF file exists
        if not os.path.exists(PDF_FILE_PATH):
            print(f"ERROR: PDF file not found at '{PDF_FILE_PATH}'.")
            print("Please ensure the file exists at the specified location.")
            return # Exit if PDF not found
        
        print(f"PDF file found: {PDF_FILE_PATH}")

        # 3. Display current index stats before indexing
        print("\n--- Index Stats Before Indexing ---")
        try:
            describe_index_stats(config.PINECONE_INDEX_NAME)
        except Exception as e:
            print(f"Could not get initial index stats (this might be normal if index doesn't exist yet): {e}")

        # 4. Run the indexing process
        print("\n--- Attempting to Index PDF ---")
        success = rag_core.index_data_from_path(PDF_FILE_PATH)

        if success:
            print("\nSUCCESS: PDF indexing process completed successfully.")
        else:
            print("\nFAILURE: PDF indexing process encountered errors or did not complete.")

        # 5. Display index stats after indexing
        print("\n--- Index Stats After Indexing ---")
        try:
            describe_index_stats(config.PINECONE_INDEX_NAME)
        except Exception as e:
            print(f"Could not get final index stats: {e}")

        # 6. Optional: Clean up (delete the index)
        if CLEAN_UP_INDEX_AFTER_TEST:
            print("\n--- Cleaning Up: Deleting Test Index ---")
            # No interactive input for automated test script, directly proceed with deletion
            # confirm_delete = input(f"Are you sure you want to delete the index '{config.PINECONE_INDEX_NAME}'? (yes/no): ").lower()
            # if confirm_delete == 'yes':
            print(f"Attempting to delete test index '{config.PINECONE_INDEX_NAME}'...")
            deleted = delete_pinecone_index(config.PINECONE_INDEX_NAME)
            if deleted:
                print(f"Test index '{config.PINECONE_INDEX_NAME}' deleted.")
            else:
                print(f"Failed to delete test index '{config.PINECONE_INDEX_NAME}'. It might have already been deleted or never created.")
            # else:
            #     print("Index deletion skipped by user input.") # Not used for auto-cleanup
        else:
            print(f"\nSkipping index cleanup. Test index '{config.PINECONE_INDEX_NAME}' was not deleted by this script.")

        print("\n--- PDF Indexing Test Finished ---")

    except ValueError as ve:
        # This will catch the ValueError from the initial config check if API keys are missing
        print(f"CRITICAL CONFIGURATION ERROR during test execution: {ve}")
        print("Please ensure your .env file is correctly set up for PINECONE_API_KEY, PINECONE_ENVIRONMENT, OPENAI_API_KEY.")
    except Exception as e:
        # Catch any other unexpected errors during the test execution
        print(f"UNEXPECTED ERROR during PDF Indexing Test: {e}")
    finally:
        # Restore original PINECONE_INDEX_NAME from config
        if original_pinecone_index_name is not None:
            config.PINECONE_INDEX_NAME = original_pinecone_index_name
        else:
            # If it wasn't there to begin with, remove the attribute we set
            if hasattr(config, 'PINECONE_INDEX_NAME'):
                delattr(config, 'PINECONE_INDEX_NAME')
        print(f"Restored original PINECONE_INDEX_NAME config (if any). Current is: {getattr(config, 'PINECONE_INDEX_NAME', 'Not Set')}")

if __name__ == "__main__":
    run_indexing_test()
