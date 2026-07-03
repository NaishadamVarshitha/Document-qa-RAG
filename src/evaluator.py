import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_utilization
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from src.config import GROQ_API_KEY, MODEL_NAME, EMBEDDING_MODEL

def evaluate_rag(question: str, answer: str, contexts: list):
    # Prepare data for RAGAS
    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
    }
    dataset = Dataset.from_dict(data)

    # Use Groq as evaluation LLM
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=MODEL_NAME,
        temperature=0
    )
    wrapped_llm = LangchainLLMWrapper(llm)

    # Use HuggingFace embeddings instead of OpenAI
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    wrapped_embeddings = LangchainEmbeddingsWrapper(embeddings)

    # Run evaluation
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_utilization],
        llm=wrapped_llm,
        embeddings=wrapped_embeddings,
    )

    return {
        "faithfulness": round(result["faithfulness"], 2),
        "answer_relevancy": round(result["answer_relevancy"], 2),
        "context_utilization": round(result["context_utilization"], 2),
    }