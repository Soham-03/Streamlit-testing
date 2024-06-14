import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.clients.quadrant import QuadrantClient
from langchain.chains.question_answering import load_qa_chain
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Quadrant client
quadrant_client = QuadrantClient(api_key=os.getenv("QUADRANT_API_KEY"))

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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def store_embeddings_in_quadrant(text_chunks):
    embeddings_generator = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    for chunk in text_chunks:
        embeddings = embeddings_generator.generate(chunk)
        quadrant_client.insert_vector(vector=embeddings, metadata={"chunk": chunk})

def user_query_processing(user_question):
    embeddings_generator = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    query_embedding = embeddings_generator.generate(user_question)
    search_results = quadrant_client.search_vector(vector=query_embedding, top_k=3)

    if search_results:
        documents = [result.metadata['chunk'] for result in search_results]
        chain = get_conversation_chain()
        response = chain({"input_documents": documents, "question": user_question}, return_only_outputs=True)
        st.write(response["output_text"], unsafe_allow_html=True)
    else:
        st.write("Sorry, I don't know the answer.")

def get_conversation_chain():
    prompt_template = """
        Answer the question clear and precise. If not provided the context return the result as
        "Sorry I don't know the answer", don't provide the wrong answer.
        Context:\n {context}?\n
        Question:\n{question}\n
        Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    chain = load_qa_chain(model, chain_type='stuff', prompt=prompt)
    return chain

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
                    if raw_text:
                        text_chunks = get_text_chunks(raw_text)
                        store_embeddings_in_quadrant(text_chunks)
                    else:
                        st.error("Failed to extract text from PDF.")
            else:
                st.error("Please upload at least one PDF file.")

    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        user_query_processing(user_question)

if __name__ == "__main__":
    main()
