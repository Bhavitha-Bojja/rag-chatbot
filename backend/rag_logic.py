import os
from dotenv import load_dotenv
from google import genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found")

CHROMA_DIR = "chroma_db"
GEN_MODEL = "gemini-3.1-flash-lite-preview"
TOKEN_WARN_THRESHOLD = 6000
TOKEN_HARD_LIMIT = 10000
MAX_CONTEXT_CHARS_PER_CHUNK = 1200


def load_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=google_api_key
    )

    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )


def build_context(results):
    context_parts = []

    for doc in results:
        source = doc.metadata.get("source", "Unknown Source")
        section = doc.metadata.get("section", "Unknown Section")
        chunk_id = doc.metadata.get("chunk_id", "Unknown Chunk")
        text = doc.page_content[:MAX_CONTEXT_CHARS_PER_CHUNK]

        context_parts.append(
            f"Source: {source} | Section: {section} | Chunk ID: {chunk_id}\n{text}"
        )

    return "\n\n".join(context_parts)


def answer_question(question, vector_store):
    results = vector_store.max_marginal_relevance_search(
        question,
        k=6,
        fetch_k=15
    )

    context = build_context(results)

    prompt = f"""
You are a helpful RAG assistant.

Answer only from the provided context.
If the context contains a partial answer, summarize what is available.
If the answer is completely missing, say:
"I could not find that in the provided documents."

Context:
{context}

Question:
{question}

Give a clear and concise answer.
"""

    client = genai.Client(api_key=google_api_key)

    token_info = client.models.count_tokens(
        model=GEN_MODEL,
        contents=prompt
    )
    estimated_input_tokens = token_info.total_tokens

    if estimated_input_tokens >= TOKEN_HARD_LIMIT:
        return "The retrieved context is too large for a safe LLM call right now. Please try a more specific question."

    response = client.models.generate_content(
        model=GEN_MODEL,
        contents=prompt
    )

    return response.text