import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.config import CHROMA_DB_PATH, EMBEDDING_MODEL, GROQ_API_KEY, MODEL_NAME, TOP_K

def build_qa_chain(collection_name: str = "default"):
    # Step 1: Load ChromaDB
    embedding_fn = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embedding_fn,
        collection_name=collection_name
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    # Step 2: Load LLM
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=MODEL_NAME,
        temperature=0
    )

    # Step 3: Prompt Template
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a helpful assistant analyzing a document.
Answer the question clearly and in detail using ONLY the context below.
If the answer is not in the context, say "I don't know based on the provided document."
You may make reasonable inferences from the context but always stay grounded in it.
Be specific and structured in your response.
Context:
{context}

Question: {question}

Answer:"""
    )

    # Step 4: Build chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever