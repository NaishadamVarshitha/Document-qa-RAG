from dotenv import load_dotenv
import os
load_dotenv()
print("API KEY:", os.getenv("GROQ_API_KEY"))
from src.chain import build_qa_chain

chain, retriever = build_qa_chain()

question = "What is the project about?"
print(f"\n🔍 Question: {question}\n")

answer = chain.invoke(question)
print(f"💬 Answer: {answer}")

# Show source chunks
print("\n📌 Source chunks used:")
docs = retriever.invoke(question)
for doc in docs:
    print(f"\nPage {doc.metadata.get('page', '?')}: {doc.page_content[:200]}...")