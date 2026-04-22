import os
from dotenv import load_dotenv
from google import genai
import chainlit as cl
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

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
            return "The retrieved context is too large for a safe LLM call right now. Please try a more specific question.", results

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
        print(f"\nGeneration failed: {e}\n")
        return "I ran into a model or quota issue while generating the answer. Please try again in a bit.", results


@cl.on_chat_start
async def start():
    vector_store = load_vector_store()
    cl.user_session.set("vector_store", vector_store)

    await cl.Message(
        content="Hi, I’m your RAG chatbot. Ask me anything from your indexed documents."
    ).send()


@cl.on_message
async def main(message: cl.Message):
    vector_store = cl.user_session.get("vector_store")
    answer, _ = answer_question(message.content, vector_store)
    await cl.Message(content=answer).send()