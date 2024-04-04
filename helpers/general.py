import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# from langchain.chains.question_answering import load_qa_chain

from config import get_settings

settings = get_settings()


def generate_response(user_question, vector_store, chat_history):
    logger = st.session_state["logger"]
    logger.info(f"User passed in question: {user_question}")

    # Do similarity search
    match = vector_store.similarity_search(user_question)

    # define the LLM
    llm = ChatOpenAI(
        openai_api_key=settings.openai_api_key,
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
            "chat_history": chat_history.messages,
        }
    )
    logger.info(f"Response: {response['answer']}")
    return response["answer"]
