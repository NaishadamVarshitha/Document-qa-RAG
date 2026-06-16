import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.1-8b-instant"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_DB_PATH = "./chroma_store"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 4