import config
import file_processor
import vector_store_manager

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA # A standard chain for RAG
from langchain.chains.llm import LLMChain # For a more custom approach if needed
from langchain_core.runnables import RunnablePassthrough # For LCEL
from langchain_core.output_parsers import StrOutputParser

# Initialize LLM
llm = None
if config.OPENAI_API_KEY:
    try:
        llm = ChatOpenAI(
            model_name=config.LLM_MODEL_NAME,
            temperature=0.3, # Adjust for creativity vs. factuality
            openai_api_key=config.OPENAI_API_KEY
        )
        print(f"LLM ({config.LLM_MODEL_NAME}) initialized successfully.")
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        llm = None
else:
    print("OpenAI API Key not found. LLM cannot be initialized. Query answering will be limited.")

def index_data_from_path(path: str):
    """Loads, processes, and indexes documents from the given path."""
    try:
        config.check_config() # Ensure API keys and index name are set
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Please set up your .env file correctly before indexing.")
        return False

    print(f"Starting data indexing from path: {path}")
    documents = file_processor.load_documents_from_path(path)
    if not documents:
        print("No documents found or loaded. Indexing aborted.")
        return False

    chunked_documents = file_processor.split_documents(documents)
    if not chunked_documents:
        print("No chunks created from documents. Indexing aborted.")
        return False

    try:
        # Get the vector store, ensuring it's recreated if it already exists
        vector_store = vector_store_manager.get_vector_store(
            index_name=config.PINECONE_INDEX_NAME,
            recreate_if_exists=True
        )
        
        # Add documents to the obtained vector store instance
        print(f"Upserting {len(chunked_documents)} chunks to index '{config.PINECONE_INDEX_NAME}'...")
        vector_store.add_documents(chunked_documents) # Using the instance method
        print("Data indexing completed successfully.")
        vector_store_manager.describe_index_stats(config.PINECONE_INDEX_NAME) # Describe stats after upsert
        return True
    except Exception as e:
        print(f"Error during data indexing: {e}")
        return False

def answer_query(user_query: str) -> str:
    """Answers a user query using the RAG pipeline."""
    if not llm:
        return "LLM not initialized. Cannot answer queries. Please check your OpenAI API key."
    if not vector_store_manager.pc:
        return "Pinecone client not initialized. Cannot retrieve context. Please check your Pinecone credentials."

    print(f"Answering query: '{user_query}'")
    try:
        # 1. Get the vector store (retriever)
        vector_store = vector_store_manager.get_vector_store(config.PINECONE_INDEX_NAME, recreate_if_exists=False)
        retriever = vector_store.as_retriever(search_kwargs={'k': 3}) # Retrieve top 3 relevant chunks

        # 2. Define a prompt template
        prompt_template = """
        You are an AI assistant. Answer the question based ONLY on the following context.
        If the context does not contain the answer, state that you don't know.
        Do not use any prior knowledge.

        Context:
        {context}

        Question: {question}

        Answer:
        """
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # 3. Create a RetrievalQA chain (a common way to do RAG)
        # This chain combines retrieval and question answering.
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # 'stuff' puts all context into one prompt
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True # Optionally return source documents
        )

        print("Retrieving context and generating answer...")
        response = qa_chain.invoke({"query": user_query}) # 'query' is the default input key for RetrievalQA

        answer = response.get("result", "Sorry, I could not generate an answer.")
        source_documents = response.get("source_documents", [])

        if source_documents:
            print("\n--- Sources Used ---")
            for i, doc in enumerate(source_documents):
                source_name = doc.metadata.get('source', 'Unknown')
                # content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                print(f"Source {i+1}: {source_name} (Page Content Snippet: {doc.page_content[:150]}...)")
            print("--------------------\n")
        else:
            print("No source documents were retrieved or returned by the chain.")

        return answer

    except Exception as e:
        print(f"Error during query answering: {e}")
        return f"An error occurred: {e}"

# --- Example Usage ---
def main():
    """Example of RAG core functions."""
    try:
        config.check_config()
    except ValueError as e:
        print(f"Configuration Error: {e}")
        print("Please ensure your .env file is set up with API keys and PINECONE_INDEX_NAME.")
        return

    # ---- Test Indexing ----
    # Create a dummy file for indexing test
    sample_data_dir = "sample_rag_data"
    os.makedirs(sample_data_dir, exist_ok=True)
    sample_file_path = os.path.join(sample_data_dir, "rag_test.txt")
    with open(sample_file_path, "w") as f:
        f.write("The capital of France is Paris. Paris is known for the Eiffel Tower.\n")
        f.write("Langchain is a framework for developing applications powered by language models.\n")
        f.write("Pinecone helps build high-performance vector search applications.")
    
    print("--- Testing Indexing ---")
    # Before running, ensure PINECONE_INDEX_NAME in .env is set.
    # You might want to delete the index in Pinecone console if re-running this often to avoid duplicate data,
    # or add logic to clear the index.
    
    # Comment out indexing if you've already indexed and don't want to re-index every time
    # print(f"Attempting to index data from: {sample_data_dir}")
    # success = index_data_from_path(sample_data_dir)
    # if success:
    #     print("Sample data indexed.")
    # else:
    #     print("Sample data indexing failed.")
    #     # If indexing fails, querying won't work well.
    #     # os.remove(sample_file_path)
    #     # os.rmdir(sample_data_dir)
    #     # return

    print("\n--- Testing Query Answering ---")
    if not llm:
        print("LLM not available, skipping query test.")
        # os.remove(sample_file_path)
        # os.rmdir(sample_data_dir)
        return

    # Query 1
    query1 = "What is the capital of France?"
    print(f"\nQuery 1: {query1}")
    answer1 = answer_query(query1)
    print(f"Answer 1: {answer1}")

    # Query 2
    query2 = "What is Langchain?"
    print(f"\nQuery 2: {query2}")
    answer2 = answer_query(query2)
    print(f"Answer 2: {answer2}")

    # Query 3 (information not in the dummy file)
    query3 = "What is the weather like today?"
    print(f"\nQuery 3: {query3}")
    answer3 = answer_query(query3)
    print(f"Answer 3: {answer3}")

    # Clean up dummy file
    # import os
    # os.remove(sample_file_path)
    # os.rmdir(sample_data_dir)

if __name__ == "__main__":
    import os # Required for os.makedirs in main, ensure it's imported if running directly
    main()
