import streamlit as st
import pandas as pd
import os
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# ---------------- CONFIG ---------------- #
st.set_page_config(page_title="AI Data Analyst Bot")
st.title("📊 AI Data Analyst Bot")

# ---------------- LOAD API ---------------- #
load_dotenv()

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

# ---------------- CLEAN CODE ---------------- #
def clean_code(code):
    code = re.sub(r"```python", "", code)
    code = code.replace("```", "")
    return code.strip()

# ---------------- FILE UPLOAD ---------------- #
file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    question = st.text_input("Ask a question about your data")

    if question:


        prompt = f"""
You are a senior data analyst.

Dataset columns: {list(df.columns)}

Write ONLY Python pandas code to answer the question.

Rules:
- Dataframe name is df
- Store final answer in variable 'result'
- No print()
- No explanation
- No markdown

Question: {question}
"""

        response = llm.invoke(prompt)

        code = clean_code(response.content)

        st.subheader("🧠 Generated Code")
        st.code(code)

        # 🔥 EXECUTION
        try:
            local_vars = {"df": df}
            exec(code, {}, local_vars)

            if "result" in local_vars:
                st.subheader("📊 Answer")
                st.write(local_vars["result"])
            else:
                st.warning("No result found.")

        except Exception as e:
            st.error(f"Execution Error: {e}")