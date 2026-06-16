import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from src.ingest import load_and_ingest
from src.chain import build_qa_chain
import tempfile
import os
import hashlib

st.set_page_config(page_title="Document Q&A", page_icon="📄")
st.title("📄 Document Q&A — RAG System")
st.markdown("Upload a PDF and ask questions about it.")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    if st.session_state.get("last_file") != uploaded_file.name:
        st.session_state.clear()
        st.session_state.last_file = uploaded_file.name
        st.session_state.chat_history = []

        collection_name = hashlib.md5(uploaded_file.name.encode()).hexdigest()[:8]
        st.session_state.collection_name = collection_name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        with st.spinner("Embedding document..."):
            load_and_ingest(tmp_path, collection_name)
            st.session_state.chain, st.session_state.retriever = build_qa_chain(collection_name)
        st.success("✅ Document ready! Ask your questions below.")

# Display chat history
if "chat_history" in st.session_state:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# Chat input
if "chain" in st.session_state:
    question = st.chat_input("Ask a question about the document...")
    if question:
        # Show user message
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.chain.invoke(question)
            st.markdown(answer)

            with st.expander("📌 Source chunks used"):
                docs = st.session_state.retriever.invoke(question)
                for doc in docs:
                    st.markdown(f"**Page {doc.metadata.get('page', '?')}:** {doc.page_content[:300]}...")

        # Save assistant response
        st.session_state.chat_history.append({"role": "assistant", "content": answer})