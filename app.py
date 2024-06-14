import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from qdrant_client import QdrantClient

# Load environment variables
load_dotenv()
GENAI_API_KEY = os.getenv("GOOGLE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    chunks = text_splitter.split_text(text)
    return chunks

def store_embeddings_in_quadrant(text_chunks):
    embeddings_generator = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    for chunk in text_chunks:
        embeddings = embeddings_generator.generate(chunk)
        # Store embeddings in Qdrant
        qdrant_client.insert_vector(vector=embeddings, metadata={"chunk": chunk})

def main():
    st.set_page_config(page_title="Chat PDF", layout="wide")
    st.header("Chat with PDF using Google's Generative AI")

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True)
        if st.button("Submit & Process PDF"):
            if pdf_docs:
                with st.spinner("Processing PDF files..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text:
                        text_chunks = get_text_chunks(raw_text)
                        store_embeddings_in_quadrant(text_chunks)
                    else:
                        st.error("Failed to extract text from PDF.")
            else:
                st.error("Please upload at least one PDF file.")

    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        # Here, implement a function to process the user's question and fetch responses
        # This function would involve interacting with the stored embeddings in Qdrant
        # and possibly using generative AI for generating responses if needed.
        pass

if __name__ == "__main__":
    main()
