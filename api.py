from fastapi import FastAPI, UploadFile, File, Form
import pandas as pd
import time
import logging

from backend import create_qa_system_from_text, create_data_agent

app = FastAPI()

# ✅ Setup logging
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

@app.get("/")
def home():
    return {"message": "GenAI API is running 🚀"}


@app.post("/ask")
async def ask_question(
    file: UploadFile = File(...),
    question: str = Form(...)
):
    start_time = time.time()

    try:
        logging.info(f"Received request | File: {file.filename} | Question: {question}")

        content = await file.read()

        # File handling
        if file.filename.endswith(".csv"):
            from io import StringIO
            df = pd.read_csv(StringIO(content.decode("utf-8")))
            qa = create_data_agent(df)

        elif file.filename.endswith(".xlsx"):
            from io import BytesIO
            df = pd.read_excel(BytesIO(content))
            qa = create_data_agent(df)

        elif file.filename.endswith(".txt"):
            text = content.decode("utf-8")
            qa = create_qa_system_from_text(text)

        elif file.filename.endswith(".pdf"):
            from PyPDF2 import PdfReader
            from io import BytesIO

            reader = PdfReader(BytesIO(content))
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""

            qa = create_qa_system_from_text(text)

        else:
            logging.error("Unsupported file type")
            return {"error": "Unsupported file type"}

        # Run QA
        answer = qa(question)

        end_time = time.time()
        duration = round(end_time - start_time, 2)

        logging.info(f"Response generated in {duration}s")

        return {
            "answer": answer,
            "response_time": f"{duration}s"
        }

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")

        return {
            "error": "Something went wrong. Please try again.",
            "details": str(e)
        }