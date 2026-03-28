from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

def create_qa_system_from_docs(documents):
    # Split documents
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Embeddings
    embeddings = OpenAIEmbeddings()

    # Vector store
    vectorstore = FAISS.from_documents(texts, embeddings)

    # LLM
    llm = ChatOpenAI()

    # QA Chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

    return qa