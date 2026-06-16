import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import CHROMA_DB_PATH, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

def load_and_ingest(pdf_path: str, collection_name: str = "default"):
    # Step 1: Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"📄 Loaded {len(documents)} pages")

    # Step 2: Filter empty pages
    documents = [doc for doc in documents if doc.page_content.strip()]
    print(f"📄 Non-empty pages: {len(documents)}")

    # Step 3: Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)

    # Step 4: Filter empty chunks
    chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
    print(f"✂️  Split into {len(chunks)} chunks")

    if not chunks:
        print("❌ No text found in PDF. It may be a scanned/image-based PDF.")
        return None

    # Step 5: Embed + store in ChromaDB
    embedding_fn = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_fn,
        persist_directory=CHROMA_DB_PATH,
        collection_name=collection_name
    )
    print(f"✅ Stored in ChromaDB at {CHROMA_DB_PATH}")
    return vectorstore