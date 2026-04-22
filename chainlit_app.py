import os
from dotenv import load_dotenv
import chainlit as cl
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

CHROMA_DIR = "chroma_db"


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


def answer_question(question, vector_store):
    results = vector_store.max_marginal_relevance_search(
        question,
        k=8,
        fetch_k=20
    )

    context = "\n\n".join(
        [
            f"Source: {doc.metadata.get('source')} | Chunk ID: {doc.metadata.get('chunk_id')}\n{doc.page_content}"
            for doc in results
        ]
    )

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

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=google_api_key,
        temperature=0.2
    )

    response = llm.invoke(prompt)
    return response.content, results


@cl.on_chat_start
async def start():
    vector_store = load_vector_store()
    cl.user_session.set("vector_store", vector_store)
    await cl.Message(
        content="Hi, I am your RAG chatbot. Ask me anything from your uploaded documents."
    ).send()


@cl.on_message
async def main(message: cl.Message):
    vector_store = cl.user_session.get("vector_store")

    answer, results = answer_question(message.content, vector_store)

    sources = "\n".join(
        [
            f"- {doc.metadata.get('source')} | Chunk {doc.metadata.get('chunk_id')}"
            for doc in results
        ]
    )

    final_response = f"{answer}\n\n**Retrieved Sources:**\n{sources}"

    await cl.Message(content=final_response).send()