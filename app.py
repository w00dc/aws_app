import streamlit as st
import os
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import ChatOpenAI
from langchain.memory import ChatMessageHistory

from config import get_settings, LoggingFormatter

settings = get_settings()

OPENAI_API_KEY = settings.openai_api_key


def read_pdf_into_chunks(file):
    # Instantiate the bulk of text to empty
    # and iterate over each file, extracting the
    # text. Separate each file with two newlines.
    text = ""
    for f in file:
        pdf_reader = PdfReader(f)
        for page in pdf_reader.pages:
            text += page.extract_text()
        text += "\n\n"

    # Break it into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n", chunk_size=1000, chunk_overlap=150, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


# Generate OpenAIEmbeddings()
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Load the vectorstore from a file if it is there
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
with st.sidebar:
    st.title("Uploaded Documents")
    file = st.file_uploader(
        " Upload PDF file(s) of rules or campaigns and ask questions",
        type="pdf",
        accept_multiple_files=True,
        help="Select multiple PDFs and the Dungeon Master Assistant will read them for fielding questions",
    )

# If files have been uploaded,
# extract the text from them
if file:
    # Read the PDF file indo chunks
    chunks = read_pdf_into_chunks(file)

    # Load the vectors if the are already present
    if not os.path.exists(settings.vectorstore_path):
        # Creating vector store - FAISS
        vector_store = FAISS.from_texts(chunks, embeddings)
        # Saving it to the path in the environment variable
        vector_store.save_local(settings.vectorstore_path)

    # Create a chat input field and store
    # the user's input in user_question
    user_question = st.chat_input("Type your question here")

    # Instantiate an empty chat history
    ephemeral_chat_history = ChatMessageHistory()

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
                # Do similarity search
                match = vector_store.similarity_search(user_question)

                # define the LLM
                llm = ChatOpenAI(
                    openai_api_key=OPENAI_API_KEY,
                    temperature=0,
                    max_tokens=1000,
                    model_name="gpt-3.5-turbo",
                )

                # stuff qa chain
                # chain -> take the question, get relevant document, pass it to the LLM, generate the output
                # chain = load_qa_chain(llm, chain_type="stuff")
                # response = chain.invoke(
                #     {
                #         "input_documents": match,
                #         "question": user_question,
                #         "messages": ephemeral_chat_history.messages,
                #     }
                # )
                # response_text = response["output_text"]

                # ConversationalRetrievalChain
                chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=vector_store.as_retriever(),
                )
                response = chain.invoke(
                    {
                        "question": user_question,
                        "chat_history": ephemeral_chat_history.messages,
                    }
                )
                response_text = response["answer"]

                st.write(response_text)

        # Add the response to the chat history
        ephemeral_chat_history.add_ai_message(response_text)

        # Add response to the session state
        message = {"role": "assistant", "content": response_text}
        st.session_state.messages.append(message)
