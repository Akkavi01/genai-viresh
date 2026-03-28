from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

import os
from dotenv import load_dotenv

load_dotenv()

def create_qa_system_from_docs(documents):
    # Split documents
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = OpenAIEmbeddings()

    # Create vector store
    vectorstore = FAISS.from_documents(texts, embeddings)

    # Create retriever
    retriever = vectorstore.as_retriever()

    # LLM
    llm = ChatOpenAI()

    # Return both
    return retriever, llm