import streamlit as st
from pyPDF2 import pdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

#extract the pdf and get the pdf details
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader=pdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

#divide the text into chunk
def get_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    chunks=text_splitter.split_text(text)
    return chunks

def get_vectore_store(text_chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectore_store=FAISS.from_texts(text_chunks,embedding=embeddings)
    vectore_store.save_local("faiss_index")

def get_convercational_chain():
    prommpt_template=  """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model=ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)

    prompt=PromptTemplate(template=prommpt_template, input_variables=["contex","question"])
    chain=load_qa_chain(model, chain_type="studff",prompt=prompt)
    return chain

