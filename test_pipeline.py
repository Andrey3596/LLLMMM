# test_pipeline.py
import json
import argparse

from topic_filter import filter_chunks_by_topic
from question_generator import (
    generate_questions_for_chunk,
    build_reference_answer,
    TYPE_LABELS,
)


def generate_test(
    embeddings_json_path: str,
    topic: str,
    question_config: dict,
    output_path: str = "questions.json",
    threshold: float = 0.8,
    model_name: str = "deepvk/USER2-base",
):
    """Полный пайплайн: тема → чанки → вопросы → questions.json"""

    # --- Шаг 1: фильтрация чанков ---
    print(f"[1/3] Поиск чанков по теме: «{topic}» (порог {threshold})")
    chunks = filter_chunks_by_topic(
        topic=topic,
        embeddings_json_path=embeddings_json_path,
        threshold=threshold,
        model_name=model_name,
    )

    if not chunks:
        print(f"[!] Не найдено чанков с похожестью >= {threshold}.")
        print("    Попробуй снизить порог или уточнить тему.")
        return []

    print(f"    Найдено чанков: {len(chunks)}")
    for c in chunks:
        print(f"      #{c['id']}  sim={c['similarity']:.4f}  «{c['text'][:60]}...»")

    # --- Шаг 2: генерация вопросов ---
    print(f"\n[2/3] Генерация вопросов, конфиг: {question_config}")
    all_questions = []

    for chunk in chunks:
        print(f"    Чанк #{chunk['id']} ...")
        qs = generate_questions_for_chunk(chunk["text"], question_config)

        for q in qs:
            q["source_chunk_id"] = chunk["id"]
            q["source_chunk_similarity"] = chunk["similarity"]
            q["topic"] = topic
            q["type_label"] = TYPE_LABELS.get(q["type"], q["type"])
            q["reference_answer"] = build_reference_answer(q, chunk["text"])

        all_questions.extend(qs)
        print(f"      → сгенерировано: {len(qs)}")

    # --- Шаг 3: сохранение ---
    print(f"\n[3/3] Сохранение в {output_path}")
    output = {
        "topic": topic,
        "threshold": threshold,
        "embeddings_file": embeddings_json_path,
        "question_config": question_config,
        "chunks_used": [
            {"id": c["id"], "similarity": c["similarity"]} for c in chunks
        ],
        "questions_count": len(all_questions),
        "questions": all_questions,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"    Готово: {len(all_questions)} вопросов → {output_path}")
    return all_questions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Генерация вопросов по теме из документа")
    parser.add_argument("--embeddings", default="doc_embeddings.json",
                        help="Путь к json с эмбеддингами чанков")
    parser.add_argument("--topic", required=True,
                        help="Тема для фильтрации чанков")
    parser.add_argument("--mc", type=int, default=2,
                        help="Сколько тестовых вопросов (выбор варианта) с каждого чанка")
    parser.add_argument("--fill", type=int, default=1,
                        help="Сколько вопросов 'вписать слово' с каждого чанка")
    parser.add_argument("--open", type=int, default=1,
                        help="Сколько вопросов с развёрнутым ответом с каждого чанка")
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="Порог похожести чанка с темой")
    parser.add_argument("--output", default="questions.json",
                        help="Куда сохранить итог")
    args = parser.parse_args()

    # Собираем конфиг из аргументов и выкидываем нулевые типы,
    # чтобы не тратить вызовы LLM зря
    config = {
        "multiple_choice": args.mc,
        "fill_in_blank": args.fill,
        "open_ended": args.open,
    }
    config = {k: v for k, v in config.items() if v > 0}

    if not config:
        print("[!] Все типы вопросов = 0. Нечего генерировать.")
        exit(1)

    generate_test(
        embeddings_json_path=args.embeddings,
        topic=args.topic,
        question_config=config,
        output_path=args.output,
        threshold=args.threshold,
    )