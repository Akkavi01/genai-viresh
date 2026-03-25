import streamlit as st
import pandas as pd
from backend import create_qa_system_from_text

st.set_page_config(page_title="GenAI Assistant", layout="wide")

st.title("📊 GenAI Assistant (Multi-File Support)")

# File uploader
uploaded_file = st.file_uploader(
    "Upload your file (CSV, Excel, TXT, PDF)",
    type=["csv", "xlsx", "txt", "pdf"]
)


# Function to extract text
def extract_text(file):
    import pandas as pd

    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
        return df.to_string()

    elif file.name.endswith(".xlsx"):
        df = pd.read_excel(file)
        return df.to_string()

    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8")

    elif file.name.endswith(".pdf"):
        from PyPDF2 import PdfReader
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

    return ""


# Main logic
if uploaded_file is not None:
    try:
        text_data = extract_text(uploaded_file)

        st.success("✅ File uploaded successfully")

        st.subheader("📄 Preview")
        st.text(text_data[:1000])

        # Create QA system
        qa_system = create_qa_system_from_text(text_data)

        st.subheader("💬 Ask Questions")

        query = st.text_input("Enter your question:")

        if query:
            with st.spinner("Thinking..."):
                response = qa_system(query)

            st.subheader("📌 Answer")
            st.write(response)

    except Exception as e:
        st.error(f"❌ Error: {e}")