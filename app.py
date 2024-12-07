import os
import base64
import time
import datetime
from dotenv import load_dotenv
import streamlit as st
from streamlit_chat import message
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


# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
hf_token=os.getenv("HF_TOKEN")

llm=ChatGroq(groq_api_key=groq_api_key, model="Llama3-8b-8192")

# Prompt Template
prompt = ChatPromptTemplate.from_template(
    """
    You are an assistant for Question-answering tasks.
    Use the following pieces of retrieved context to answer
    the question. If you don't know the answer, say that you
    don't know. Use three sentences maximum and keep the
    answer concise, short, and to the point. You are an expert in adventure and activities like Land, Water, and Air. 
    Our focus region is India, not beyond that. 
    Remember source location is Pune, India.

    <context>
    {context}
    <context>
    Question: {input}
    """
)

# Set page configuration
st.set_page_config(layout="wide", page_title="AdvenBuddy Chatbot")

# Background styling
with open("deep_ocean.jpg", "rb") as f:
    background = f.read()

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/jpg;base64,{base64.b64encode(background).decode()});
        background-size: cover;
        color: white;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Title
st.title("🌊 AdvenBuddy: Your Personalized Assistant for Adventures")

# Initialize session state for embeddings and chat history
if "vectors" not in st.session_state:
    st.session_state.vectors = None
    st.session_state.embeddings = None
    st.session_state.chat_history = []  # To store user-bot chat messages

# Function to create vector embeddings
def create_vector_embeddings():
    if st.session_state.vectors is None:
        st.session_state.embeddings = HuggingFaceEmbeddings()
        loader = PyPDFLoader("advenbuddy_rag_pdfs_demo.pdf")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs[:50])
        st.session_state.vectors = FAISS.from_documents(final_documents, st.session_state.embeddings)
        st.success("Vector Database is Ready!")

# Chat interface
user_input = st.text_input("Enter your question:")

# Button to create document embeddings
if st.button("Create Embeddings"):
    create_vector_embeddings()

# Process user input
if user_input:
    if st.session_state.vectors:
        # Create document chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Generate response
        start = time.process_time()
        response = retrieval_chain.invoke({"input": user_input})
        bot_reply = response["answer"]
        st.session_state.chat_history.append({"user": user_input, "bot": bot_reply})
        #st.markdown(f"<p style='font-size: 20px;'>{bot_reply}</p>", unsafe_allow_html=True)

        # Debug: Print response time
        print(f"Response time: {time.process_time() - start} seconds")

        # Display similar documents in expander
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(f"**Document {i+1}**")
                st.write(doc.page_content)
                st.write("------")
    else:
        st.error("Please create embeddings first!")

# Display chat messages
for i, chat in enumerate(st.session_state.chat_history):
    message(chat["user"], is_user=True, key=f"user_{i}")
    message(chat["bot"], key=f"bot_{i}")
