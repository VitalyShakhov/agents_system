import os
import re
import json
from langchain_openai import ChatOpenAI

def extract_json_from_text(text: str) -> dict | None:
    """Извлекает JSON из текста (даже если он в блоках ```json)."""
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)
    # print("JSON Data")
    # print(text)
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_str = text[start:end+1]
            return json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        return None
    return None

def get_llm(temperature: float) -> ChatOpenAI:
    return ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        temperature=temperature
    )

def sanitize_filename(name: str) -> str:
    """Преобразует название в безопасное имя папки."""
    name = name.lower()
    name = re.sub(r'[^\w\s-]', '', name)
    name = re.sub(r'[\s_-]+', '_', name)
    name = name.strip('_')
    return name[:50]  # Ограничение длины


