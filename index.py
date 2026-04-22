import os
import time
import uuid
from pathlib import Path

import fitz
from docx import Document
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

DATA_DIR = Path("data")
CHROMA_DIR = "chroma_db"


def read_pdf(file_path: Path) -> str:
    text = []
    pdf = fitz.open(file_path)
    for page in pdf:
        text.append(page.get_text())
    pdf.close()
    return "\n".join(text)


def read_docx(file_path: Path) -> str:
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])


def load_documents():
    documents = []

    for file_path in DATA_DIR.iterdir():
        if file_path.suffix.lower() == ".pdf":
            text = read_pdf(file_path)
            documents.append({"source": file_path.name, "text": text})
        elif file_path.suffix.lower() == ".docx":
            text = read_docx(file_path)
            documents.append({"source": file_path.name, "text": text})

    return documents


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )

    texts = []
    metadatas = []

    for doc in documents:
        split_texts = splitter.split_text(doc["text"])
        for i, chunk in enumerate(split_texts):
            texts.append(chunk)
            metadatas.append({
                "source": doc["source"],
                "chunk_id": i
            })

    return texts, metadatas


def create_vector_store(texts, metadatas):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=google_api_key
    )

    vector_store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

    batch_size = 20
    delay_seconds = 45
    total = len(texts)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)

        batch_texts = texts[start:end]
        batch_metadatas = metadatas[start:end]
        batch_ids = [str(uuid.uuid4()) for _ in batch_texts]

        print(f"Adding chunks {start + 1} to {end} of {total}...")

        vector_store.add_texts(
            texts=batch_texts,
            metadatas=batch_metadatas,
            ids=batch_ids
        )

        if end < total:
            print(f"Waiting {delay_seconds} seconds before next batch...")
            time.sleep(delay_seconds)

    print("Chroma vector database created successfully.")


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} document(s)")

    texts, metadatas = chunk_documents(docs)
    print(f"Created {len(texts)} chunk(s)")

    create_vector_store(texts, metadatas)