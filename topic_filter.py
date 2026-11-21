# topic_filter.py
import json
import numpy as np
from sentence_transformers import SentenceTransformer


def load_embeddings_json(path):
    """Читает doc_embeddings.json → (texts, embeddings, ids)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data["items"]
    texts = [it["text"] for it in items]
    embeddings = np.array([it["embedding"] for it in items], dtype=np.float32)
    ids = [it["id"] for it in items]
    return texts, embeddings, ids


def filter_chunks_by_topic(
    topic: str,
    embeddings_json_path: str = "doc_embeddings.json",
    threshold: float = 0.8,
    prefix: str = "search_query: ",
    model_name: str = "deepvk/USER2-base",
):
    """
    Возвращает список чанков, схожих с темой >= threshold.
    Каждый элемент: {"id": int, "text": str, "similarity": float}
    Отсортировано по убыванию similarity.
    """
    model = SentenceTransformer(model_name)
    texts, doc_embs, ids = load_embeddings_json(embeddings_json_path)

    # Вектор темы (префикс search_query — как в примере из твоего исходника)
    topic_vec = model.encode(prefix + topic, normalize_embeddings=True)

    # Косинусное сходство = скалярное произведение нормализованных векторов
    sims = doc_embs @ topic_vec

    selected = []
    for i, sim in enumerate(sims):
        if float(sim) >= threshold:
            selected.append({
                "id": ids[i],
                "text": texts[i],
                "similarity": float(sim),
            })

    selected.sort(key=lambda x: x["similarity"], reverse=True)
    return selected


if __name__ == "__main__":
    TOPIC = "Моделирование"          # ← тема, которую ищем
    THRESHOLD = 0.8                  # ← порог сходства
    FILE = "doc_embeddings.json"     # ← откуда читаем

    chunks = filter_chunks_by_topic(TOPIC, FILE, THRESHOLD)

    print(f"Тема: «{TOPIC}», порог: {THRESHOLD}")
    print(f"Найдено чанков: {len(chunks)}\n")
    for c in chunks:
        print(f"  #{c['id']}  sim={c['similarity']:.4f}")
        print(f"    {c['text'][:120]}...")