import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import numpy as np

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  # Ensuring that None doesn't break concatenation
    return text

def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    embeddings = [embedding_model.embed(chunk) for chunk in text_chunks if chunk.strip()]
    
    if not embeddings or not all(isinstance(e, np.ndarray) for e in embeddings):
        st.error("Failed to generate valid embeddings. Check the embedding process.")
        return
    
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local("faiss_index")
    st.success("FAISS index created successfully!")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    the provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n{context}\n
    Question:\n{question}\n

    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config(page_title="Chat PDF", layout="wide")
    st.header("Chat with PDF using Gemini")

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True)
        if st.button("Submit & Process PDF"):
            if pdf_docs:
                with st.spinner("Processing PDF files..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_chunks(raw_text)
                    get_vector_store(text_chunks)
            else:
                st.error("Please upload at least one PDF file.")

    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()
