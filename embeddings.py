# embeddings.py
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os

model = SentenceTransformer("deepvk/USER2-base")


def to_text(data):
    if data is None:
        return ""
    if isinstance(data, str):
        return data.strip()
    if isinstance(data, (int, float, bool)):
        return str(data)
    if isinstance(data, (list, tuple, set)):
        return " ".join(to_text(item) for item in data if item)
    if isinstance(data, dict):
        parts = []
        for key, value in data.items():
            parts.append(f"{key}: {to_text(value)}")
        return " ".join(parts)
    return str(data)


def read_file(path):
    ext = os.path.splitext(path)[1].lower()

    if ext in (".txt", ".md", ".csv"):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return to_text(data)

    if ext == ".pdf":
        try:
            import pymupdf
            doc = pymupdf.open(path)
            return "\n".join(page.get_text() for page in doc)
        except ImportError:
            raise ImportError("Установите: pip install pymupdf")

    if ext in (".docx", ".doc"):
        try:
            from docx import Document
            doc = Document(path)
            return "\n".join(p.text for p in doc.paragraphs)
        except ImportError:
            raise ImportError("Установите: pip install python-docx")

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def chunk_text(text, chunk_size=1000, overlap=100):
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if end < len(text):
            last_dot = max(chunk.rfind(". "), chunk.rfind("! "), chunk.rfind("? "))
            if last_dot > chunk_size * 0.5:
                chunk = chunk[:last_dot + 1]
                end = start + last_dot + 1
        chunks.append(chunk.strip())
        start = end - overlap

    return [c for c in chunks if c]


def get_embeddings(data, prefix="search_document: ", chunk=True):
    if isinstance(data, str) and os.path.isfile(data):
        text = read_file(data)
    else:
        text = to_text(data)

    if chunk:
        texts = chunk_text(text)
    else:
        texts = [text]

    prefixed = [prefix + t for t in texts]

    embeddings = model.encode(
        prefixed,
        normalize_embeddings=True,
        batch_size=8,
        show_progress_bar=True,
    )

    return texts, embeddings


def save_embeddings_json(texts, embeddings, path="embeddings.json"):
    data = {
        "model": "deepvk/USER2-base",
        "count": len(texts),
        "dim": int(embeddings.shape[1]),
        "items": [
            {
                "id": i,
                "text": text,
                "embedding": vec.tolist(),
            }
            for i, (text, vec) in enumerate(zip(texts, embeddings))
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"JSON сохранён: {path}")


if __name__ == "__main__":
    FILE = "test.docx"                              
    OUTPUT = "doc_embeddings.json"
    texts, emb = get_embeddings(FILE)
    print(f"Чанков: {len(texts)}, размерность: {emb.shape[1]}")
    save_embeddings_json(texts, emb, OUTPUT)