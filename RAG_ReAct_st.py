# RAG streamlit aap

import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader,UnstructuredHTMLLoader,WebBaseLoader,JSONLoader,PyPDFLoader,UnstructuredPDFLoader,UnstructuredWordDocumentLoader, Docx2txtLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from openai import OpenAI
import sys
# from langchain.document_loaders import DocxReader



# Load API Key from file
file_path = "C:\\Users\\raxx7\\Documents\\Python_Scripts\\rag\\.openai_api_key.txt"
if os.path.exists(file_path):
    with open(file_path) as f:
        OPENAI_API_KEY = f.read().strip()
        # Check if the key is not empty
        if not OPENAI_API_KEY:
            st.error("API key file is empty")
            st.stop()
        model = OpenAI(api_key=OPENAI_API_KEY)
else:
    print(f"File not found: {file_path}")
    st.error("API Key file not found")
    st.stop()

# Initialize session state for storing the conversation
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Load documents
loader2 = DirectoryLoader('C:/Users/raxx7/Documents/Python_Scripts/llm/openAI/datainformation/pdf2', glob="**/*.pdf",
                        show_progress=True, loader_cls=PyPDFLoader)

docs_pdf = loader2.load()

# 2. Load Word Documents
# loader_word = UnstructuredWordDocumentLoader(
#     'C:/Users/raxx7/Documents/Python_Scripts/llm/openAI/datainformation/word',
#     glob="**/*.docx", 
#     show_progress=True)
# docs_word = loader_word.load()

# 3. Load Web Content
urls = [
    "https://calteches.library.caltech.edu/3200/1/moon.pdf",
    "https://en.wikipedia.org/wiki/Black_hole"
    # Add your URLs here
]
loader_web = WebBaseLoader(urls)
docs_web = loader_web.load()

# Combine all documents
all_docs = docs_pdf + docs_web

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(all_docs)

# Initialize the Chroma vector store
CHROMA_PATH = "./chroma_db_combined"
embedding_model_openai = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
db_chroma = Chroma.from_documents(
    chunks, 
    embedding_model_openai, 
    collection_name="vector_database",
    persist_directory=CHROMA_PATH
)
retriever = db_chroma.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# The rest of your code remains the same...
# Define the ReAct prompt template
REACT_PROMPT = """
You are a helpful assistant. Below is the system context information. Your task is to answer the user's question based only on the information provided.

### SYSTEM CONTEXT:
{context}

### USER QUESTION:
{question}

### INSTRUCTIONS:
- First, reason through the question by breaking it down and analyzing it step by step.
- Once your reasoning is complete, provide a clear and concise answer.
- Base your answer ONLY on the information provided in the context.
- Do not include information not mentioned in the context.
- Do not use phrases like "according to the context" or "based on the information provided."
- Ensure your answer is detailed and comprehensive.

### REASONING:
{reasoning}

### FINAL ANSWER:
"""

prompt_template = REACT_PROMPT
chat_model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)

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

#### ---------------------------------------------------------------------------------------------------
# Define the ReAct prompt template
# REACT_PROMPT = """
# You are a helpful assistant. Below is the system context information. Your task is to answer the user's question based only on the information provided.

# ### SYSTEM CONTEXT:
# {context}

# ### USER QUESTION:
# {question}

# ### INSTRUCTIONS:
# - First, reason through the question by breaking it down and analyzing it step by step.
# - Once your reasoning is complete, provide a clear and concise answer.
# - Base your answer ONLY on the information provided in the context.
# - Do not include information not mentioned in the context.
# - Do not use phrases like "according to the context" or "based on the information provided."
# - Ensure your answer is detailed and comprehensive.

# ### REASONING:
# Let me think through this question carefully...
# """

# # Create the LLM
# chat_model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)

# # Function to format documents into a string
# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# # ReAct RAG function
# def react_rag(question):
#     # Step 1: Retrieve relevant documents
#     docs = retriever.get_relevant_documents(question)
#     context = format_docs(docs)
    
#     # Step 2: Create messages with the ReAct prompt
#     messages = [
#         {"role": "system", "content": "You are a helpful assistant that answers questions based only on provided context."},
#         {"role": "user", "content": REACT_PROMPT.format(context=context, question=question)}
#     ]
    
#     # Step 3: Generate response using the chat model
#     response = chat_model.invoke(messages)
    
#     # Return the content of the response
#     return response.content

# # Streamlit Web App
# st.title("AI Conversational Bot - ReAct RAG")

# # Display conversation history first
# for msg in st.session_state["messages"]:
#     st.chat_message(msg["role"]).write(msg["content"])

# # Input field at the bottom
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
#         # Get AI response using ReAct RAG
#         response_content = react_rag(user_input)
        
#         # Add assistant's response to history and display it
#         st.session_state["messages"].append({"role": "assistant", "content": response_content})
#         st.chat_message("assistant").write(response_content)
#     except Exception as e:
#         st.error(f"Error processing input: {str(e)}")

# # Simple instruction at the bottom
# st.caption("To exit, type 'bye', 'quit', or 'exit'.")




### new method -----------------------------------------------------------------



# # Define the ReAct prompt template
# REACT_PROMPT = """
# You are a helpful assistant. Below is the system context information. Your task is to answer the user's question based only on the information provided.

# ### SYSTEM CONTEXT:
# {context}

# ### USER QUESTION:
# {question}

# ### INSTRUCTIONS:
# - First, reason through the question by breaking it down and analyzing it step by step.
# - Once your reasoning is complete, provide a clear and concise answer.
# - Base your answer ONLY on the information provided in the context.
# - Do not include information not mentioned in the context.
# - Do not use phrases like "according to the context" or "based on the information provided."
# - Ensure your answer is detailed and comprehensive.

# ### REASONING:
# {reasoning}

# ### FINAL ANSWER:
# """

# prompt_template = REACT_PROMPT
# chat_model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)

def reasoning_and_answering(context, question, prompt_template, chat_model):
    # Format the prompt with context, question, and initial reasoning
    reasoning = "Let me analyze this question step by step based on the provided context..."
    
    messages = [
        {"role": "system", "content": prompt_template.format(
            context=context, 
            question=question,
            reasoning=reasoning
        )}
    ]
    
    # Get the final answer from the model
    response = chat_model.invoke(messages)
    return response.content

def create_rag_chain_with_react(retriever, prompt_template, chat_model):
    # Function to format documents into a string
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Chain: retrieve context, reason, then answer the question
    def process_question(question):
        # Retrieve relevant documents
        docs = retriever.get_relevant_documents(question)
        context = format_docs(docs)
        
        # Get reasoned answer
        final_answer = reasoning_and_answering(context, question, prompt_template, chat_model)
        return final_answer
    
    return process_question

# Create the RAG chain
rag_chain_with_react = create_rag_chain_with_react(retriever, prompt_template, chat_model)

# Streamlit Web App
st.title("AI Conversational Bot - ReAct RAG")

# Display conversation history first
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# Input field at the bottom
user_input = st.chat_input("Ask a question based on the PDF documents...")

# Process user input if provided
if user_input:
    # Check if user wants to exit
    if user_input.lower() in ['exit', 'bye', 'quit']:
        st.write("**Exiting...**")
        sys.exit()
    
    # Add user message to history and display it
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)
    
    try:
        # Get AI response using ReAct RAG
        response_content = rag_chain_with_react(user_input)
        
        # Add assistant's response to history and display it
        st.session_state["messages"].append({"role": "assistant", "content": response_content})
        st.chat_message("assistant").write(response_content)
    except Exception as e:
        st.error(f"Error processing input: {str(e)}")

# Simple instruction at the bottom
st.caption("To exit, type 'bye', 'quit', or 'exit'.")