import streamlit as st
import os
import logging
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ChatMessageHistory

from config import get_settings, create_logger
from helpers import read_pdf_into_chunks, get_files_in_path, generate_response

settings = get_settings()


# Initialize logger
if "logger" not in st.session_state:
    st.session_state["logger"] = create_logger(
        name=f"{settings.app_name}",
        level=f"{settings.log_level}",
        file=f"{settings.app_name}.log",
    )
logger = st.session_state["logger"]

# Initialize an empty chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = ChatMessageHistory()
ephemeral_chat_history = st.session_state["chat_history"]

# Generate OpenAIEmbeddings()
embeddings = OpenAIEmbeddings(openai_api_key=settings.openai_api_key)


# Load the vectorstore from a file if it is there
# TODO: Need to link the logic here up to the file
#       uploader.
if os.path.exists(settings.vectorstore_path):
    # To load vector_store from a saved file...
    vector_store = FAISS.load_local(
        settings.vectorstore_path, embeddings, allow_dangerous_deserialization=True
    )

# Title of the main pane
st.header("Dungeon Master Assistant")

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I help you?"}
    ]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Configure the sidebar for uploading
# PDF document(s)
# TODO: Replace this with a folder of PDFs to ingest automatically???
#       Change this sidebar so that it can toggle between multiple
#       Subdirs of PDF collections, each with their own vector stores
# with st.sidebar:
#     st.title("Uploaded Documents")
#     files = st.file_uploader(
#         " Upload PDF file(s) of rules or campaigns and ask questions",
#         type="pdf",
#         accept_multiple_files=True,
#         help="Select multiple PDFs and the Dungeon Master Assistant will read them for fielding questions",
#     )

# Set a list a files from a pre-set directory
if "files" not in st.session_state.keys():
    st.session_state.files = get_files_in_path(settings.files_path)
files = st.session_state.files

# If files have be instantiated...
if files:
    # Load the vectors if the are already present
    if not os.path.exists(settings.vectorstore_path):
        # Read the PDF file into chunks
        chunks = read_pdf_into_chunks(files)
        # Creating vector store - FAISS
        vector_store = FAISS.from_texts(chunks, embeddings)
        # Saving it to the path in the environment variable
        # TODO: After we modify this to pull PDFs from
        #       a directory/other place, change this to store
        #       it's vector store somewhere, accordingly
        vector_store.save_local(settings.vectorstore_path)

    # Create a chat input field and store
    # the user's input in user_question
    user_question = st.chat_input("Type your question here")

    # If the user has input a question...
    if user_question:
        # Add the user's question to the chat history
        ephemeral_chat_history.add_user_message(user_question)

        # Add it to the session state
        st.session_state.messages.append({"role": "user", "content": user_question})
        # Write the user's question to the chat
        with st.chat_message("user"):
            st.write(user_question)

        # Make a spinner while the bot
        # generates it's response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Generate the response
                response_text = generate_response(
                    user_question, vector_store, ephemeral_chat_history
                )
                # Write out the response text
                st.write(response_text)

        # Add the response to the chat history
        ephemeral_chat_history.add_ai_message(response_text)

        # Add response to the session state
        message = {"role": "assistant", "content": response_text}
        st.session_state.messages.append(message)
        logger.debug(f"Chat history: {ephemeral_chat_history.messages}")
