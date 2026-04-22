import os
import re
import uuid
import time
from pathlib import Path

import fitz
import numpy as np
from docx import Document
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

DATA_DIR = Path("data")
CHROMA_DIR = "chroma_db"

# Local model only for semantic merge during chunking
SEMANTIC_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


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
        elif file_path.suffix.lower() == ".docx":
            text = read_docx(file_path)
        else:
            continue

        documents.append({
            "source": file_path.name,
            "text": text
        })

    return documents


def is_heading(line: str) -> bool:
    line = line.strip()

    if not line:
        return False

    if re.match(r"^\d+(\.\d+)*\s+.+", line):
        return True

    if len(line.split()) <= 12 and line[0].isupper():
        if not line.endswith("."):
            return True

    return False


def split_into_sections(text: str):
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]

    sections = []
    current_title = "Untitled Section"
    current_paragraphs = []

    for line in lines:
        if is_heading(line):
            if current_paragraphs:
                sections.append({
                    "section": current_title,
                    "paragraphs": current_paragraphs
                })
                current_paragraphs = []
            current_title = line
        else:
            current_paragraphs.append(line)

    if current_paragraphs:
        sections.append({
            "section": current_title,
            "paragraphs": current_paragraphs
        })

    return sections


def build_paragraph_blocks(paragraphs, block_max_chars=700):
    """
    First pass: create small paragraph blocks.
    These are smaller than final chunks.
    """
    blocks = []
    current_block = []

    for para in paragraphs:
        candidate = "\n".join(current_block + [para])

        if len(candidate) <= block_max_chars:
            current_block.append(para)
        else:
            if current_block:
                blocks.append("\n".join(current_block).strip())
            current_block = [para]

    if current_block:
        blocks.append("\n".join(current_block).strip())

    return blocks


def cosine_similarity(vec_a, vec_b):
    a = vec_a / np.linalg.norm(vec_a)
    b = vec_b / np.linalg.norm(vec_b)
    return float(np.dot(a, b))


def semantic_merge_blocks(blocks, semantic_model, max_chars=1800, similarity_threshold=0.68):
    """
    Second pass: merge neighboring blocks if they are semantically related
    and the merged size stays under max_chars.
    """
    if not blocks:
        return []

    merged_chunks = []
    i = 0

    while i < len(blocks):
        current_text = blocks[i]
        j = i + 1

        while j < len(blocks):
            candidate_merge = current_text + "\n\n" + blocks[j]

            if len(candidate_merge) > max_chars:
                break

            embeddings = semantic_model.encode(
                [current_text, blocks[j]],
                convert_to_numpy=True
            )

            sim = cosine_similarity(embeddings[0], embeddings[1])

            if sim >= similarity_threshold:
                current_text = candidate_merge
                j += 1
            else:
                break

        merged_chunks.append(current_text)
        i = j

    return merged_chunks


def build_structured_semantic_chunks(documents, semantic_model):
    texts = []
    metadatas = []

    for doc in documents:
        sections = split_into_sections(doc["text"])

        for section_index, section in enumerate(sections):
            blocks = build_paragraph_blocks(
                section["paragraphs"],
                block_max_chars=700
            )

            final_chunks = semantic_merge_blocks(
                blocks,
                semantic_model=semantic_model,
                max_chars=1800,
                similarity_threshold=0.68
            )

            for chunk_index, chunk_text in enumerate(final_chunks):
                texts.append(chunk_text)
                metadatas.append({
                    "source": doc["source"],
                    "section": section["section"],
                    "section_id": section_index,
                    "chunk_id": chunk_index,
                    "doc_chunk_uid": str(uuid.uuid4())
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
        batch_ids = [meta["doc_chunk_uid"] for meta in batch_metadatas]

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
    semantic_model = SentenceTransformer(SEMANTIC_MODEL_NAME)

    docs = load_documents()
    print(f"Loaded {len(docs)} document(s)")

    texts, metadatas = build_structured_semantic_chunks(docs, semantic_model)
    print(f"Built {len(texts)} structured-semantic chunk(s)")

    create_vector_store(texts, metadatas)