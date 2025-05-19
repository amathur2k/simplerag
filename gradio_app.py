import gradio as gr
import os
import io
import sys
from contextlib import redirect_stdout

# Langchain & Pinecone components for re-initialization
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone

import config
import rag_core
import vector_store_manager
from vector_store_manager import describe_index_stats as vsm_describe_index_stats
from vector_store_manager import delete_pinecone_index as vsm_delete_pinecone_index
from vector_store_manager import pc as initial_pinecone_client # Keep track of initial for comparison or re-init


# --- GUI Specific Functions ---

def get_active_embedding_model_name():
    """Utility to get the name of the currently configured embedding model."""
    if vector_store_manager.embeddings_model:
        if isinstance(vector_store_manager.embeddings_model, OpenAIEmbeddings):
            return getattr(vector_store_manager.embeddings_model, 'model', 'OpenAI Embeddings (model name N/A)')
        elif isinstance(vector_store_manager.embeddings_model, HuggingFaceEmbeddings):
            return getattr(vector_store_manager.embeddings_model, 'model_name', 'HuggingFace Embeddings (model name N/A)')
    return "N/A (Not Initialized)"

def gui_check_config_on_load():
    """Checks configuration on GUI load and returns a status message."""
    status_messages = []
    try:
        config.check_config() # This will raise ValueError if something is missing
        status_messages.append(f"Initial Configuration Check: OK")
        status_messages.append(f"Pinecone Index Name: {config.PINECONE_INDEX_NAME}")
        status_messages.append(f"Pinecone Environment/Region: {config.PINECONE_ENVIRONMENT}")
        status_messages.append(f"LLM Model: {config.LLM_MODEL_NAME}")
        status_messages.append(f"Embedding Model Active: {get_active_embedding_model_name()}")

        if not rag_core.llm:
            status_messages.append("WARNING: LLM (OpenAI) not initialized. Querying will not work. Check OPENAI_API_KEY.")
        if not vector_store_manager.pc:
            status_messages.append("WARNING: Pinecone client not initialized. Operations may fail. Check PINECONE_API_KEY & PINECONE_ENVIRONMENT.")
        
        return "\n".join(status_messages)
    except ValueError as e:
        return f"CRITICAL CONFIGURATION ERROR: {e}\nPlease set required API keys and settings and 'Apply & Reload Configuration' or update .env and restart."
    except Exception as e:
        return f"An unexpected error occurred during configuration check: {e}"

def gui_index_data(path_input: str):
    """Handles indexing data from a given path via the GUI."""
    if not path_input:
        return "Path cannot be empty. Please provide a valid file or directory path."
    if not os.path.exists(path_input):
        return f"Error: Path '{path_input}' does not exist on this system."
    
    f = io.StringIO()
    success = False
    try:
        with redirect_stdout(f):
            # rag_core.index_data_from_path prints progress and returns a boolean
            success = rag_core.index_data_from_path(path_input)
        printed_output = f.getvalue().strip()
        
        if success:
            return f"{printed_output}\n\nIndexing process completed successfully for '{path_input}'."
        else:
            # The printed_output should contain reasons for failure from rag_core
            return f"{printed_output}\n\nIndexing process failed or encountered errors for '{path_input}'. Check messages above."
    except Exception as e:
        # Catch any unexpected errors during the call
        printed_output = f.getvalue().strip()
        return f"{printed_output}\n\nAn unexpected error occurred during indexing: {e}".strip()

def gui_ask_question(query_input: str):
    """Handles asking a question via the GUI."""
    if not query_input:
        return "Question cannot be empty. Please enter your query."
    
    f = io.StringIO()
    answer = ""
    try:
        with redirect_stdout(f):
            # rag_core.answer_query prints sources and returns the answer string
            answer = rag_core.answer_query(query_input)
        printed_output = f.getvalue().strip()
        
        # Combine printed sources (if any) with the answer
        if printed_output and printed_output not in answer: # Avoid duplicating if answer includes prints
            return f"{printed_output}\n\nAnswer:\n{answer}".strip()
        return f"Answer:\n{answer}" # Default if no separate prints or if answer contains them
    except Exception as e:
        printed_output = f.getvalue().strip()
        return f"{printed_output}\n\nAn error occurred while processing your question: {e}".strip()

def gui_view_stats():
    """Handles viewing Pinecone index statistics via the GUI."""
    try:
        config.check_config() # Basic check for config values
        if not vector_store_manager.pc: # Check the re-initializable client
            return "Pinecone client not initialized. Cannot fetch stats. Check Pinecone API key/environment."
        
        index_name = config.PINECONE_INDEX_NAME
        # vsm_describe_index_stats (from vector_store_manager) prints to console and returns a stats object.
        # We want to format the object cleanly for the GUI.
        stats_obj = vsm_describe_index_stats(index_name)

        if stats_obj: # stats_obj is a pinecone.generated.data.models.IndexStats object
            formatted_stats = f"Statistics for Pinecone Index: '{index_name}'\n"
            formatted_stats += f"---------------------------------------\n"
            formatted_stats += f"  Total Vectors : {getattr(stats_obj, 'total_vector_count', 'N/A')}\n"
            formatted_stats += f"  Dimensions    : {getattr(stats_obj, 'dimension', 'N/A')}\n"
            
            namespaces_data = getattr(stats_obj, 'namespaces', None)
            if namespaces_data:
                formatted_stats += "  Namespaces (name: vector_count):\n"
                for ns_name, ns_data in namespaces_data.items():
                    formatted_stats += f"    - \"{ns_name}\": {getattr(ns_data, 'vector_count', 'N/A')} vectors\n"
            else:
                formatted_stats += "  Namespaces    : No namespace data available or index might be empty.\n"
            return formatted_stats
        else:
            # If stats_obj is None, vsm_describe_index_stats might have failed and printed an error to its console.
            return f"Could not retrieve stats for index '{index_name}'. The function 'describe_index_stats' in vector_store_manager might have printed details to the application console."

    except ValueError as e: # From config.check_config()
        return f"Configuration Error: {e}"
    except ConnectionError as e: # If Pinecone client isn't properly initialized
        return f"Pinecone Connection Error: {e}"
    except Exception as e:
        return f"An unexpected error occurred while fetching index stats: {e}"

def gui_delete_index(confirm_delete: bool):
    """Handles deleting the Pinecone index via the GUI with confirmation."""
    if not confirm_delete:
        return "Deletion not confirmed. Index was NOT deleted."
    
    try:
        config.check_config()
        if not vector_store_manager.pc: # Check the re-initializable client
            return "Pinecone client not initialized. Cannot delete index. Check Pinecone API key/environment."

        index_name = config.PINECONE_INDEX_NAME
        f = io.StringIO()
        deleted = False
        error_during_delete = False
        with redirect_stdout(f):
            try:
                # vsm_delete_pinecone_index prints status and returns boolean
                # suppress_confirm=True because Gradio checkbox handles confirmation
                deleted = vsm_delete_pinecone_index(index_name, suppress_confirm=True) 
            except Exception as e_delete:
                print(f"Error during the 'delete_pinecone_index' call: {e_delete}") # Captured by StringIO
                error_during_delete = True
        
        printed_output = f.getvalue().strip()

        if error_during_delete:
            return printed_output # Contains the specific error message from the delete function call

        if deleted:
            result_message = f"Index '{index_name}' has been successfully deleted."
        else:
            result_message = f"Failed to delete index '{index_name}' or it did not exist. See messages above or application console for details."
        
        # Prepend any informative prints from the delete function itself
        if printed_output and printed_output not in result_message:
             return f"{printed_output}\n{result_message}".strip()
        return result_message

    except ValueError as e: # From config.check_config()
        return f"Configuration Error: {e}"
    except ConnectionError as e: # If Pinecone client isn't properly initialized
        return f"Pinecone Connection Error: {e}"
    except Exception as e:
        return f"An unexpected error occurred during the index deletion process: {e}"

# --- Configuration Update Function ---
def gui_set_and_reload_config(openai_key_in, pinecone_key_in, pinecone_env_in, pinecone_index_in):
    """Updates os.environ, config module, and re-initializes services."""
    results = []    
    os.environ['OPENAI_API_KEY'] = openai_key_in
    os.environ['PINECONE_API_KEY'] = pinecone_key_in
    os.environ['PINECONE_ENVIRONMENT'] = pinecone_env_in
    os.environ['PINECONE_INDEX_NAME'] = pinecone_index_in
    results.append("Environment variables updated.")

    # Update config module attributes directly
    config.OPENAI_API_KEY = openai_key_in
    config.PINECONE_API_KEY = pinecone_key_in
    config.PINECONE_ENVIRONMENT = pinecone_env_in
    config.PINECONE_INDEX_NAME = pinecone_index_in
    results.append("Config module attributes updated.")

    # Re-initialize LLM in rag_core
    try:
        if config.OPENAI_API_KEY:
            rag_core.llm = ChatOpenAI(
                model_name=config.LLM_MODEL_NAME,
                temperature=0.3,
                openai_api_key=config.OPENAI_API_KEY
            )
            results.append(f"LLM ({config.LLM_MODEL_NAME}) re-initialized.")
        else:
            rag_core.llm = None
            results.append("OpenAI API Key not provided. LLM is not active.")
    except Exception as e:
        rag_core.llm = None
        results.append(f"Error re-initializing LLM: {e}")

    # Re-initialize Pinecone client in vector_store_manager
    try:
        if config.PINECONE_API_KEY and config.PINECONE_ENVIRONMENT:
            vector_store_manager.pc = Pinecone(api_key=config.PINECONE_API_KEY)
            results.append("Pinecone client re-initialized.")
        else:
            vector_store_manager.pc = None
            results.append("Pinecone credentials missing. Pinecone client not active.")
    except Exception as e:
        vector_store_manager.pc = None
        results.append(f"Error re-initializing Pinecone client: {e}")

    # Re-initialize embeddings_model in vector_store_manager
    try:
        if config.OPENAI_API_KEY:
            vector_store_manager.embeddings_model = OpenAIEmbeddings(
                model="text-embedding-ada-002",
                openai_api_key=config.OPENAI_API_KEY
            )
        else:
            vector_store_manager.embeddings_model = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
        results.append(f"Embeddings model re-initialized to: {get_active_embedding_model_name()}.")
    except Exception as e:
        vector_store_manager.embeddings_model = None
        results.append(f"Error re-initializing embeddings model: {e}")

    # Get updated overall status
    overall_status = gui_check_config_on_load()
    return "\n".join(results), overall_status


# --- Gradio Interface Definition ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="orange"), title="Simple RAG GUI") as demo:
    gr.Markdown("# Simple RAG Application GUI")

    with gr.Row():
        with gr.Column(scale=1, min_width=350): # Sidebar Column
            gr.Markdown("### Settings")
            with gr.Accordion("API & Pinecone Configuration", open=True):
                openai_key_input = gr.Textbox(label="OpenAI API Key", value=lambda: config.OPENAI_API_KEY, type="password", placeholder="sk-...")
                pinecone_key_input = gr.Textbox(label="Pinecone API Key", value=lambda: config.PINECONE_API_KEY, type="password", placeholder="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx")
                pinecone_env_input = gr.Textbox(label="Pinecone Environment/Region", value=lambda: config.PINECONE_ENVIRONMENT, placeholder="e.g., us-east-1")
                pinecone_index_input = gr.Textbox(label="Pinecone Index Name", value=lambda: config.PINECONE_INDEX_NAME, placeholder="my-index")
                
                set_config_button = gr.Button("Apply & Reload Configuration", variant="primary")
                config_set_status_output = gr.Textbox(label="Configuration Update Status", interactive=False, lines=5, max_lines=10)
            
            gr.Markdown("--- System Status ---")
            system_status_output = gr.Textbox(
                label="Current System & Configuration Status", 
                value=gui_check_config_on_load, 
                interactive=False, 
                lines=8, 
                max_lines=15,
                show_copy_button=True
            )
            # Link button action to update both status boxes
            set_config_button.click(
                fn=gui_set_and_reload_config, 
                inputs=[openai_key_input, pinecone_key_input, pinecone_env_input, pinecone_index_input],
                outputs=[config_set_status_output, system_status_output]
            )

        with gr.Column(scale=3): # Main Content Column
            with gr.Tabs():
                with gr.TabItem("1. Index Content"):
                    gr.Markdown("### Index Files or Directory\nEnter the full path to a local file or a directory to process and index its contents into Pinecone.")
                    index_path_input = gr.Textbox(label="File/Directory Path", placeholder="e.g., C:\\MyData or /home/user/data")
                    index_button = gr.Button("Start Indexing", variant="primary")
                    index_status_output = gr.Textbox(label="Indexing Status & Logs", interactive=False, lines=10, max_lines=20, show_copy_button=True)
                    index_button.click(fn=gui_index_data, inputs=index_path_input, outputs=index_status_output)

                with gr.TabItem("2. Ask a Question"):
                    gr.Markdown("### Ask a Question\nQuery your indexed knowledge base. The answer will be generated based on the relevant content found.")
                    query_input_text = gr.Textbox(label="Your Question", placeholder="e.g., What are the key features of product X?")
                    query_button = gr.Button("Get Answer", variant="primary")
                    answer_output_text = gr.Textbox(label="Answer & Sources", interactive=False, lines=15, max_lines=30, show_copy_button=True)
                    query_button.click(fn=gui_ask_question, inputs=query_input_text, outputs=answer_output_text)

                with gr.TabItem("3. View Index Statistics"):
                    gr.Markdown(f"### Pinecone Index Statistics") # Index name will be in system_status_output
                    stats_button = gr.Button("Refresh Index Stats", variant="secondary")
                    stats_output_text = gr.Textbox(label="Index Statistics", interactive=False, lines=10, max_lines=15, show_copy_button=True)
                    # Load stats when app starts/tab is first viewed or system status changes
                    demo.load(fn=gui_view_stats, inputs=None, outputs=stats_output_text)
                    system_status_output.change(fn=gui_view_stats, inputs=None, outputs=stats_output_text) # Refresh stats if config changes
                    stats_button.click(fn=gui_view_stats, inputs=None, outputs=stats_output_text)

                with gr.TabItem("4. Delete Pinecone Index"):
                    gr.Markdown("### Delete Pinecone Index (CAUTION!)") # Index name in system_status_output
                    gr.Markdown(
                        f"**DANGER ZONE**: This action will permanently delete the entire Pinecone index specified in the configuration. "
                        "This cannot be undone. Ensure you have selected the correct index and understand the consequences."
                    )
                    # The label for checkbox will now be generic as index name is in status
                    confirm_delete_checkbox = gr.Checkbox(label=f"I confirm I want to permanently delete the currently configured Pinecone index.")
                    delete_button = gr.Button("Delete Index Now", variant="stop")
                    delete_status_output = gr.Textbox(label="Deletion Status", interactive=False, lines=5, max_lines=10, show_copy_button=True)
                    delete_button.click(fn=gui_delete_index, inputs=confirm_delete_checkbox, outputs=delete_status_output)

if __name__ == "__main__":
    print("Launching Simple RAG Gradio GUI...")
    print("Please wait for the Gradio interface to start. It usually runs on http://127.0.0.1:7860")
    print("You can open this URL in your web browser.")
    demo.launch() # Launches on a local URL by default
