from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import CHROMA_DB_PATH, EMBEDDING_MODEL, TOP_K

# Load existing ChromaDB
embedding_fn = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = Chroma(
    persist_directory=CHROMA_DB_PATH,
    embedding_function=embedding_fn
)

# Test query
query = "What is the project about?"
results = vectorstore.similarity_search(query, k=TOP_K)

print(f"\n🔍 Query: {query}")
print(f"📦 Top {TOP_K} chunks retrieved:\n")
for i, doc in enumerate(results):
    print(f"--- Chunk {i+1} (Page {doc.metadata.get('page', '?')}) ---")
    print(doc.page_content[:300])
    print()