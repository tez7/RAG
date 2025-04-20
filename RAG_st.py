# RAG streamlit aap

import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader,UnstructuredHTMLLoader,WebBaseLoader,JSONLoader,PyPDFLoader,UnstructuredPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from openai import OpenAI
import sys


# # Load API Key from file
# file_path = "C:\\Users\\raxx7\\Documents\\Python_Scripts\\rag\\.openai_api_key.txt"
# if os.path.exists(file_path):
#     with open(file_path) as f:
#         OPENAI_API_KEY = f.read().strip()
#         # Check if the key is not empty
#         if not OPENAI_API_KEY:
#             st.error("API key file is empty")
#             st.stop()
#         model = OpenAI(api_key=OPENAI_API_KEY)
# else:
#     print(f"File not found: {file_path}")
#     st.error("API Key file not found")
#     st.stop()

# # Initialize session state for storing the conversation
# if "messages" not in st.session_state:
#     st.session_state["messages"] = []

# # Load documents
# loader2 = DirectoryLoader('C:/Users/raxx7/Documents/Python_Scripts/llm/openAI/datainformation/pdf2', glob="**/*.pdf",
#                         show_progress=True, loader_cls=PyPDFLoader)

# docs2 = loader2.load()

# # Split documents into chunks
# text_splitter2 = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# chunks2 = text_splitter2.split_documents(docs2)

# # Initialize the Chroma vector store
# CHROMA_PATH = "./chroma_db3_"
# embedding_model_openai = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
# db_chroma = Chroma.from_documents(
#     chunks2, 
#     embedding_model_openai, 
#     collection_name="vector_database",
#     persist_directory=CHROMA_PATH
# )
# retriever = db_chroma.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# # Define the prompt template
# PROMPT_TEMPLATE2 = """
# You are a helpful assistant. Below is the system context information. Your task is to answer the user's question based only on the information provided.

# ### SYSTEM CONTEXT:
# {context}

# ### USER QUESTION:
# {question}

# ### INSTRUCTIONS:
# - Answer the question based only on the above system context.
# - Provide a detailed answer.
# - Do not include information not mentioned in the context.
# - Avoid justifying your answers or giving additional information.
# - Do not say "according to the context", "mentioned in the context", or similar phrases in your response.
# - Ensure your answer is entirely derived from the provided context.
# """

# prompt_template2 = ChatPromptTemplate.from_template(PROMPT_TEMPLATE2)
# chat_model2 = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# rag_chain2 = {"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt_template2 | chat_model2

# # Streamlit Web App
# st.title("AI Conversational Bot - RAG")

# # Display conversation history first
# for msg in st.session_state["messages"]:
#     st.chat_message(msg["role"]).write(msg["content"])

# # Input field at the bottom using chat_input() instead of text_input()
# user_input = st.chat_input("Ask a question based on the PDF documents...")

# # Process user input if provided
# if user_input:
#     # Check if user wants to exit
#     if user_input.lower() in ['exit', 'bye', 'quit']:
#         st.write("**Exiting...**")
#         sys.exit()
        
#     # Add user message to history and display it
#     st.session_state["messages"].append({"role": "user", "content": user_input})
#     st.chat_message("user").write(user_input)
    
#     try:
#         # Get AI response from RAG chain
#         response = rag_chain2.invoke(user_input)
#         response_content = response.content
        
#         # Add assistant's response to history and display it
#         st.session_state["messages"].append({"role": "assistant", "content": response_content})
#         st.chat_message("assistant").write(response_content)
#     except Exception as e:
#         st.error(f"Error processing input: {str(e)}")

# # Simple instruction at the bottom
# st.caption("To exit, type 'bye', 'quit', or 'exit'.")










import os
import sys
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from openai import OpenAI
import streamlit as st

# Initialize Streamlit app
st.set_page_config(page_title="AI Conversational Bot - RAG")

# Load OpenAI API Key
@st.cache_resource
def load_api_key():
    file_path = "C:\\Users\\raxx7\\Documents\\Python_Scripts\\rag\\.openai_api_key.txt"
    if os.path.exists(file_path):
        with open(file_path) as f:
            OPENAI_API_KEY = f.read().strip()
            if not OPENAI_API_KEY:
                st.error("API key file is empty")
                st.stop()
            return OPENAI_API_KEY
    else:
        st.error("API Key file not found")
        st.stop()

OPENAI_API_KEY = load_api_key()
model = OpenAI(api_key=OPENAI_API_KEY)

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Define prompt template
PROMPT_TEMPLATE2 = """
You are a helpful assistant. Below is the system context information. Your task is to answer the user's question based only on the information provided.

### SYSTEM CONTEXT:
{context}

### USER QUESTION:
{question}

### INSTRUCTIONS:
- Answer the question based only on the above system context.
- Provide a detailed answer.
- Do not include information not mentioned in the context.
- Avoid justifying your answers or giving additional information.
- Do not say "according to the context", "mentioned in the context", or similar phrases in your response.
- Ensure your answer is entirely derived from the provided context.
"""

# Initialize RAG system with caching
@st.cache_resource
def initialize_rag_system():
    # Load documents
    loader2 = DirectoryLoader(
        'C:/Users/raxx7/Documents/Python_Scripts/llm/openAI/datainformation/pdf2',
        glob="**/*.pdf",
        show_progress=True,
        loader_cls=PyPDFLoader
    )
    docs2 = loader2.load()

    # Split documents
    text_splitter2 = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks2 = text_splitter2.split_documents(docs2)

    # Create vector store
    CHROMA_PATH = "./chroma_db3_"
    embedding_model_openai = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db_chroma = Chroma.from_documents(
        chunks2,
        embedding_model_openai,
        collection_name="vector_database",
        persist_directory=CHROMA_PATH
    )

    # Create retriever
    retriever = db_chroma.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Create prompt template
    prompt_template2 = ChatPromptTemplate.from_template(PROMPT_TEMPLATE2)
    chat_model2 = ChatOpenAI(api_key=OPENAI_API_KEY)

    # Format documents function
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Create RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template2
        | chat_model2
    )
    
    return rag_chain

# Initialize the RAG system
rag_chain2 = initialize_rag_system()

# Streamlit UI
st.title("AI Conversational Bot - RAG")

# Display conversation history
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# Chat input
user_input = st.chat_input("Ask a question based on the PDF documents...")

# Process user input
if user_input:
    # Check for exit commands
    if user_input.lower() in ['exit', 'bye', 'quit']:
        st.write("Goodbye! 👋")
        st.stop()
        
    # Add user message to history
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)
    
    try:
        # Get AI response
        response = rag_chain2.invoke(user_input)
        response_content = response.content
        
        # Add assistant response to history
        st.session_state["messages"].append({"role": "assistant", "content": response_content})
        st.chat_message("assistant").write(response_content)
    except Exception as e:
        st.error(f"Error processing your question: {str(e)}")


# What This Does
# @st.cache_resource Decorator:

# Tells Streamlit to cache the result of this function

# Only runs the function once, even if the script reruns

# Ideal for expensive operations like loading documents or creating vector stores

# The initialize_rag() Function:

# Contains all your initialization code (document loading, splitting, vector DB creation)

# Returns your fully configured rag_chain2

# How It Works:

# First run: Executes the full initialization

# Subsequent runs: Returns the cached result instead of recomputing