# question_generator.py
import json
import re
import requests


# ===== НАСТРОЙКА LLM (Ollama + Qwen) =====
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:14b"   # можно поменять на qwen2.5:14b, если есть 16+ ГБ RAM


def call_llm(prompt: str, system_prompt: str = "") -> str:
    """Отправляет запрос в Ollama и возвращает текст ответа."""
    full_prompt = (system_prompt + "\n\n" + prompt).strip()
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": full_prompt,
            "stream": False,
            "options": {"temperature": 0.4},   # 0.4 — чтобы не слишком фантазировала
        },
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()["response"]


# ===== ПРОМПТЫ ПОД КАЖДЫЙ ТИП =====

SYSTEM_PROMPT = (
    "Ты — опытный методист. Строго следуй формату ответа. "
    "Используй ТОЛЬКО информацию из текста, не придумывай."
)

PROMPTS = {
    "multiple_choice": """Составь {count} тестовых вопроса (с выбором одного правильного ответа) по тексту ниже.
У каждого вопроса должно быть 4 варианта (A, B, C, D), только один правильный.
Верни СТРОГО JSON-массив без пояснений:
[
  {{"question": "текст вопроса", "options": ["A) ...", "B) ...", "C) ...", "D) ..."], "correct": "A"}}
]
Текст:
{text}""",

    "fill_in_blank": """Составь {count} вопросов типа «вписать пропущенное слово» по тексту ниже.
В каждом вопросе замени одно ключевое слово на "_____".
Верни СТРОГО JSON-массив без пояснений:
[
  {{"question": "текст с _____ вместо слова", "answer": "пропущенное слово"}}
]
Текст:
{text}""",

    "open_ended": """Составь {count} вопросов с развёрнутым ответом по тексту ниже.
Вопросы должны проверять понимание, а не пересказ.
Верни СТРОГО JSON-массив без пояснений:
[
  {{"question": "текст вопроса"}}
]
Текст:
{text}""",
}

TYPE_LABELS = {
    "multiple_choice": "тестовый (выбор варианта)",
    "fill_in_blank": "вписать слово",
    "open_ended": "развёрнутый ответ",
}


# ===== ПАРСИНГ ОТВЕТА LLM =====

def _extract_json(raw: str):
    """Достаёт JSON-массив из ответа LLM, даже если она обернула его в ```json ... ```."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"LLM не вернул JSON-массив:\n{raw[:500]}")
    return json.loads(raw[start:end + 1])


# ===== ГЕНЕРАЦИЯ ВОПРОСОВ ДЛЯ ОДНОГО ЧАНКА =====

def generate_questions_for_chunk(chunk_text: str, question_config: dict) -> list[dict]:
    """
    question_config: {"multiple_choice": 3, "fill_in_blank": 2, "open_ended": 1}
    Возвращает список вопросов для одного чанка.
    """
    all_questions = []
    for qtype, count in question_config.items():
        if count <= 0:
            continue

        prompt = PROMPTS[qtype].format(count=count, text=chunk_text)
        raw = call_llm(prompt, system_prompt=SYSTEM_PROMPT)

        try:
            items = _extract_json(raw)
        except Exception as e:
            print(f"[WARN] Не удалось распарсить ответ LLM ({qtype}): {e}")
            continue

        items = items[:count]   # если LLM сделала больше — обрежем
        for item in items:
            item["type"] = qtype
        all_questions.extend(items)

    return all_questions


# ===== ЭВРИСТИКА ЭТАЛОННОГО ОТВЕТА =====

def build_reference_answer(question: dict, chunk_text: str) -> str:
    """Строит эталонный ответ по простым правилам."""
    qtype = question.get("type")

    if qtype == "multiple_choice":
        letter = question.get("correct", "").strip().upper()[:1]
        for opt in question.get("options", []):
            if opt.strip().upper().startswith(letter):
                return opt
        return ""

    if qtype == "fill_in_blank":
        return question.get("answer", "")

    if qtype == "open_ended":
        # Первое предложение из чанка, где есть значимое слово из вопроса
        q_words = set(re.findall(r"\w{5,}", question.get("question", "").lower()))
        sentences = re.split(r"(?<=[.!?])\s+", chunk_text)
        for sent in sentences:
            if any(w in sent.lower() for w in q_words):
                return sent.strip()
        return sentences[0].strip() if sentences else chunk_text[:300]

    return ""


# ===== ПРОСТОЙ ТЕСТ ПРИ ЗАПУСКЕ ФАЙЛА =====

if __name__ == "__main__":
    sample_chunk = (
        "Моделирование — это метод исследования объектов на их моделях. "
        "Модель — это упрощённое представление реального объекта. "
        "Моделирование бывает материальным и идеальным."
    )
    config = {"multiple_choice": 2, "fill_in_blank": 1, "open_ended": 1}

    print(f"Тип-конфиг: {config}")
    questions = generate_questions_for_chunk(sample_chunk, config)

    for q in questions:
        q["reference_answer"] = build_reference_answer(q, sample_chunk)
        print()
        print(f"[{q['type']}] {q['question']}")
        if "options" in q:
            for opt in q["options"]:
                print(f"   {opt}")
            print(f"   ✓ correct: {q['correct']}")
        if "answer" in q:
            print(f"   ✓ пропущено: {q['answer']}")
        print(f"   → эталон: {q['reference_answer']}")