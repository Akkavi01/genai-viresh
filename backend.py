import os
from dotenv import load_dotenv

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Load API key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def create_qa_system_from_text(text):

    # Split text into chunks
    splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    docs = splitter.create_documents([text])

    # Create embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Store in vector DB
    db = FAISS.from_documents(docs, embeddings)

    # Retriever
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )

    # QA function
    def ask(question):
        docs = retriever.invoke(question)
        context = "\n".join([doc.page_content for doc in docs])

        prompt = f"""
        You are a senior business analyst.

        Use the context to answer the question.

        Context:
        {context}

        Question:
        {question}

        Answer in this format:
        - Key Insight
        - Explanation
        - Recommendation
        """

        response = llm.invoke(prompt)
        return response.content

    return ask


import pandas as pd

def create_data_agent(df):

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )

    # Basic statistics
    summary = df.describe().to_string()

    # Correlation (if Sales exists)
    corr_text = ""
    if "Sales" in df.columns:
        corr = df.corr(numeric_only=True)["Sales"].sort_values(ascending=False)
        corr_text = "Correlation with Sales:\n" + corr.to_string()

    context = summary + "\n\n" + corr_text

    def ask(question):
        prompt = f"""
        You are a senior Marketing Mix Modeling (MMM) expert.

        Use data-driven reasoning.

        Dataset Summary:
        {context}

        Question:
        {question}

        Provide:
        - Key Insight
        - Explanation
        - Recommendation
        """

        response = llm.invoke(prompt)
        return response.content

    return ask