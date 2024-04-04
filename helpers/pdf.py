import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def read_pdf_into_chunks(files):
    logger = st.session_state["logger"]
    # Instantiate the bulk of text to empty
    # and iterate over each file, extracting the
    # text. Separate each file with two newlines.
    text = ""
    for f in files:
        pdf_reader = PdfReader(f)
        for page in pdf_reader.pages:
            text += page.extract_text()
        text += "\n\n"

    # Break it into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n", chunk_size=1000, chunk_overlap=150, length_function=len
    )
    chunks = text_splitter.split_text(text)
    logger.info(f"Text in PDF split into {len(chunks)} chunks")
    return chunks
