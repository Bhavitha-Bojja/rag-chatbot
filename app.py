import os
from dotenv import load_dotenv
from google import genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

CHROMA_DIR = "chroma_db"
GEN_MODEL = "gemini-3.1-flash-lite-preview"

# Your own safety thresholds
TOKEN_WARN_THRESHOLD = 6000
TOKEN_HARD_LIMIT = 10000

# Limit how much of each chunk is sent to the LLM
MAX_CONTEXT_CHARS_PER_CHUNK = 1200


def load_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=google_api_key
    )

    vector_store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )
    return vector_store


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
    # Use MMR so retrieved chunks are more diverse and less repetitive
    results = vector_store.max_marginal_relevance_search(
        question,
        k=6,
        fetch_k=15
    )

    print("\n================ RETRIEVED CHUNK PREVIEWS ================\n")
    for i, doc in enumerate(results, 1):
        print(f"--- Result {i} ---")
        print(f"Source: {doc.metadata.get('source')}")
        print(f"Section: {doc.metadata.get('section')}")
        print(f"Chunk ID: {doc.metadata.get('chunk_id')}")
        print(doc.page_content[:700])
        print()

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

    print("\n================ CONTEXT SENT TO LLM ================\n")
    print(context)
    print("\n================ END CONTEXT ================\n")

    print("\n================ FULL PROMPT SENT TO LLM ================\n")
    print(prompt)
    print("\n================ END PROMPT ================\n")

    client = genai.Client(api_key=google_api_key)

    try:
        token_info = client.models.count_tokens(
            model=GEN_MODEL,
            contents=prompt
        )
        estimated_input_tokens = token_info.total_tokens
        print(f"\nEstimated input tokens before generation: {estimated_input_tokens}\n")

        if estimated_input_tokens >= TOKEN_HARD_LIMIT:
            return (
                f"Skipped LLM call: prompt too large "
                f"({estimated_input_tokens} tokens >= hard limit {TOKEN_HARD_LIMIT}).",
                results
            )

        if estimated_input_tokens >= TOKEN_WARN_THRESHOLD:
            print(
                f"Warning: prompt is large "
                f"({estimated_input_tokens} tokens >= warning threshold {TOKEN_WARN_THRESHOLD}).\n"
            )

    except Exception as e:
        print(f"\nToken counting failed: {e}\n")

    try:
        response = client.models.generate_content(
            model=GEN_MODEL,
            contents=prompt
        )

        print("\n================ USAGE METADATA ================\n")
        print(response.usage_metadata)
        print("\n================ END USAGE METADATA ================\n")

        return response.text, results

    except Exception as e:
        return f"Gemini request failed: {str(e)}", results


if __name__ == "__main__":
    vector_store = load_vector_store()

    question = input("Enter your question: ")
    answer, results = answer_question(question, vector_store)

    print("\n================ FINAL ANSWER ================\n")
    print(answer)

    print("\n================ RETRIEVED SOURCES ================\n")
    for i, doc in enumerate(results, 1):
        print(
            f"{i}. {doc.metadata.get('source')} | "
            f"{doc.metadata.get('section')} | "
            f"Chunk {doc.metadata.get('chunk_id')}"
        )