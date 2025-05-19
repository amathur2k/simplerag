import os
import rag_core
import config # To potentially check config or show index name
from vector_store_manager import describe_index_stats, delete_pinecone_index # For advanced options

def main_menu():
    """Displays the main menu and handles user input."""
    print("\n===================================")
    print(" Simple RAG Application Menu")
    print("===================================")
    print(f"Using Pinecone Index: {config.PINECONE_INDEX_NAME}")
    print("-----------------------------------")
    print("1. Index Files/Directory")
    print("2. Ask a Question")
    print("3. View Index Stats")
    print("4. Delete Pinecone Index (Caution!)")
    print("5. Exit")
    print("-----------------------------------")

    while True:
        choice = input("Enter your choice (1-5): ")
        if choice in ["1", "2", "3", "4", "5"]:
            return choice
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")

def handle_indexing():
    """Handles the file/directory indexing process."""
    print("\n--- Index Data ---")
    path = input("Enter the full path to the file or directory to index: ").strip()
    if not path:
        print("Path cannot be empty.")
        return
    if not os.path.exists(path):
        print(f"Error: Path '{path}' does not exist.")
        return
    
    print(f"Starting indexing for path: {path}")
    success = rag_core.index_data_from_path(path)
    if success:
        print("Indexing process completed.")
    else:
        print("Indexing process encountered errors or no data was indexed.")

def handle_querying():
    """Handles the question answering process."""
    print("\n--- Ask a Question ---")
    query = input("Enter your question: ").strip()
    if not query:
        print("Question cannot be empty.")
        return

    answer = rag_core.answer_query(query)
    print("\n--- Answer ---")
    print(answer)
    print("--------------")

def handle_view_stats():
    """Handles viewing Pinecone index statistics."""
    print("\n--- Pinecone Index Statistics ---")
    try:
        config.check_config() # Ensure Pinecone client can be initialized
        describe_index_stats(config.PINECONE_INDEX_NAME)
    except ValueError as e:
        print(f"Configuration error: {e}")
    except Exception as e:
        print(f"Error fetching index stats: {e}")

def handle_delete_index():
    """Handles deleting the Pinecone index."""
    print("\n--- Delete Pinecone Index ---")
    confirm = input(f"CAUTION: Are you sure you want to delete the ENTIRE Pinecone index '{config.PINECONE_INDEX_NAME}'? This cannot be undone. (yes/no): ").strip().lower()
    if confirm == 'yes':
        try:
            config.check_config()
            print(f"Attempting to delete index: {config.PINECONE_INDEX_NAME}")
            deleted = delete_pinecone_index(config.PINECONE_INDEX_NAME)
            if deleted:
                print(f"Index '{config.PINECONE_INDEX_NAME}' has been deleted.")
            else:
                print(f"Failed to delete index '{config.PINECONE_INDEX_NAME}' or index did not exist.")
        except ValueError as e:
            print(f"Configuration error: {e}")
        except Exception as e:
            print(f"Error deleting index: {e}")
    else:
        print("Index deletion cancelled.")

def run_application():
    """Main loop for the RAG application."""
    try:
        config.check_config() # Initial check when app starts
        print("Application configuration seems OK.")
    except ValueError as e:
        print(f"CRITICAL CONFIGURATION ERROR: {e}")
        print("Please set up your .env file with API keys and Pinecone details (PINECONE_API_KEY, PINECONE_ENVIRONMENT, OPENAI_API_KEY, PINECONE_INDEX_NAME).")
        print("The application may not function correctly until this is resolved.")
        # Optionally, exit if config is critical
        # return 

    while True:
        choice = main_menu()

        if choice == '1':
            handle_indexing()
        elif choice == '2':
            handle_querying()
        elif choice == '3':
            handle_view_stats()
        elif choice == '4':
            handle_delete_index()
        elif choice == '5':
            print("Exiting application. Goodbye!")
            break
        
        input("\nPress Enter to return to the menu...")

if __name__ == "__main__":
    run_application()
