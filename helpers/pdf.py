import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from os import listdir
from os.path import isfile, join

from config import get_settings

settings = get_settings()


def get_files_in_path(path):
    logger = st.session_state["logger"]

    onlyfiles = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    logger.debug(f"Files in path {path}:\n{onlyfiles}")
    return onlyfiles


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
        separators="\n",
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    logger.debug(f"Text in PDF split into {len(chunks)} chunks")
    return chunks
