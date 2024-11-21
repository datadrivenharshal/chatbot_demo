import os
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema import StrOutputParser
from langchain_community.vectorstores import FAISS
import base64
# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
hf_token=os.getenv("HF_TOKEN")

llm=ChatGroq(groq_api_key=groq_api_key, model="Llama3-8b-8192")



## Prompt Template
prompt = ChatPromptTemplate.from_template(
    """
    You are an assistant for Question-answering tasks.
    use the following pieces of retrieved context to answer
    the question. If you don't know the answer, say that you 
    don't know. use three sentences maximum and keep the 
    answer concise, short and to the point. You are expert in Adventure and activies like Land, Water and Air Our focus region is India not beyond that. 
    Remember Source location is Pune, India.

    <context>
    {context}
    <context>
    Question:{input}


    """
)
# Set page configuration
st.set_page_config(layout="wide")

# Load the background image
with open("deep_ocean.jpg", "rb") as f:
    background = f.read()

# Set the background image style
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/jpg;base64,{base64.b64encode(background).decode()});
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)
st.title("AdvenBuddy: your Personalized assistant for Adventures..")

def create_vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=HuggingFaceEmbeddings()
        st.session_state.loader=PyPDFLoader("D:\\chatbot_demo\\advenbuddy_rag_pdfs_demo.pdf")
        st.session_state.docs=st.session_state.loader.load()
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)


user_prompt=st.text_input("Enter your Question: ")

if st.button("Document Embedding"):
    create_vector_embeddings()
    st.write("Vector Database is Ready")

import time

if user_prompt:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)

    start=time.process_time()
    response=retrieval_chain.invoke({'input':user_prompt})
    st.markdown(f"<p style='color: white; font-size: 40px; font-family: Arial;'>{response['answer']}</p>", unsafe_allow_html=True)
    print(f"Response time : {time.process_time()-start}")

    
    
    with st.expander("Document similarity Search"):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("-------------------")
            