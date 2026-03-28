from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma

import os


def create_qa_system_from_docs(documents):
    try:
        # 🔍 Debug: Check API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is missing!")

        print("✅ API Key loaded")

        # 📄 Split documents
        text_splitter = CharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        texts = text_splitter.split_documents(documents)

        print(f"✅ Split into {len(texts)} chunks")

        # 🔗 Embeddings
        embeddings = OpenAIEmbeddings()

        # 📦 Vector Store (FAISS)
        vectorstore = Chroma.from_documents(texts, embeddings)

        print("✅ Vector store created")

        # 🔍 Retriever
        retriever = vectorstore.as_retriever()

        # 🤖 LLM
        llm = ChatOpenAI(
            temperature=0,
            model="gpt-3.5-turbo"   # safe default
        )

        print("✅ LLM initialized")

        return retriever, llm

    except Exception as e:
        print("❌ ERROR in backend:", str(e))
        raise e