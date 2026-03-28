import streamlit as st
from backend import create_qa_system_from_docs
from langchain.document_loaders import PyPDFLoader, TextLoader
import tempfile
import os

st.set_page_config(page_title="GenAI Multi-Doc Q&A")

st.title("📄 GenAI Multi-Document Q&A")

# Upload files
uploaded_files = st.file_uploader(
    "Upload documents (PDF or TXT)",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    documents = []

    for file in uploaded_files:
        try:
            # Save temp file
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name

            # Load document
            if file.name.endswith(".pdf"):
                loader = PyPDFLoader(tmp_path)
            else:
                loader = TextLoader(tmp_path)

            docs = loader.load()
            documents.extend(docs)

            # Clean temp file
            os.remove(tmp_path)

        except Exception as e:
            st.error(f"Error loading {file.name}: {e}")

    if st.button("Process Documents"):
        try:
            qa_chain = create_qa_system_from_docs(documents)
            st.session_state.qa_chain = qa_chain
            st.success("Documents processed successfully!")
        except Exception as e:
            st.error(f"Processing error: {e}")

# Q&A Section
if "qa_chain" in st.session_state:
    st.subheader("Ask Questions")

    query = st.text_input("Enter your question:")

    if st.button("Get Answer"):
        if query.strip() == "":
            st.warning("Please enter a question.")
        else:
            try:
                response = st.session_state.qa_chain.run(query)
                st.write("### Answer:")
                st.write(response)
            except Exception as e:
                st.error(f"Error: {e}")