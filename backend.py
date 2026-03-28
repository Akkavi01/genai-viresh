from langchain_openai import ChatOpenAI
import os


def create_qa_system_from_docs(documents):
    try:
        # ✅ Validate API Key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is missing!")

        print("✅ API Key loaded")

        # 📄 Combine all documents into one context
        full_text = "\n\n".join([doc.page_content for doc in documents])

        print(f"✅ Documents combined. Length: {len(full_text)}")

        # 🤖 Initialize LLM
        llm = ChatOpenAI(
            temperature=0,
            model="gpt-3.5-turbo"
        )

        print("✅ LLM initialized")

        # 🔍 Simple QA function (no vector DB)
        def qa_system(query):
            try:
                prompt = f"""
You are a helpful assistant. Answer the question using the context below.

Context:
{full_text}

Question:
{query}

Answer:
"""

                response = llm.invoke(prompt)

                print("✅ Response generated")

                return response.content

            except Exception as e:
                print("❌ Error during QA:", str(e))
                return f"Error: {str(e)}"

        return qa_system

    except Exception as e:
        print("❌ Backend initialization error:", str(e))
        raise e