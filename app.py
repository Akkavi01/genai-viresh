import streamlit as st
from backend import create_qa_system_from_docs
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import tempfile
import os

st.set_page_config(page_title="GenAI Multi-Doc Q&A")

st.title("📄 GenAI Multi-Document Q&A")

uploaded_files = st.file_uploader(
    "Upload documents (PDF or TXT)",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

documents = []

if uploaded_files:
    for file in uploaded_files:
        try:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name

            if file.name.endswith(".pdf"):
                loader = PyPDFLoader(tmp_path)
            else:
                loader = TextLoader(tmp_path)

            docs = loader.load()
            documents.extend(docs)

            os.remove(tmp_path)

        except Exception as e:
            st.error(f"Error loading {file.name}: {e}")

    if st.button("Process Documents"):
        try:
            retriever, llm = create_qa_system_from_docs(documents)
            st.session_state.retriever = retriever
            st.session_state.llm = llm
            st.success("Documents processed successfully!")
        except Exception as e:
            st.error(f"Processing error: {e}")

# Q&A
if "retriever" in st.session_state:
    st.subheader("Ask Questions")

    query = st.text_input("Enter your question:")

    if st.button("Get Answer"):
        try:
            docs = st.session_state.retriever.invoke(query)

            context = "\n\n".join([doc.page_content for doc in docs])

            prompt = f"""
            Answer the question based on the context below.

            Context:
            {context}

            Question:
            {query}
            """

            response = st.session_state.llm.invoke(prompt)

            st.write("### Answer:")
            st.write(response.content)

        except Exception as e:
            st.error(f"Error: {e}")