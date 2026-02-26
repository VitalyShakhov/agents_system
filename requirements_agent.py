import os
import json
import re
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field, ValidationError
from langchain_openai import ChatOpenAI

load_dotenv()

# === Модель требований ===
class Requirements(BaseModel):
    goal: str = Field(description="Цель приложения")
    features: list[str] = Field(description="Список основных функций")
    audience: str = Field(description="Целевая аудитория")
    special_requirements: str = Field(description="Особые условия")

# === Инициализация модели ===
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0.3
)

# === Промпт ===
prompt = ChatPromptTemplate.from_messages([
    ("system", """
Ты — аналитик требований. Пользователь описывает идею приложения простыми словами.
Если информации недостаточно для заполнения всех полей (цель, функции, аудитория, особые условия),
задай **один конкретный уточняющий вопрос**.

Если информации достаточно — **выведи ТОЛЬКО ЧИСТЫЙ JSON в формате**:
{{
  "goal": "...",
  "features": ["...", "..."],
  "audience": "...",
  "special_requirements": "..."
}}

Без пояснений, без markdown, без ```json — только валидный JSON.
"""),
    MessagesPlaceholder(variable_name="chat_history"),
])

chat_chain = prompt | llm

def extract_json_from_text(text: str) -> dict | None:
    """Пытается извлечь JSON из текста (даже если он в блоках ```json)."""
    # Удаляем markdown-блоки
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)
    try:
        # Ищем первую фигурную скобку до последней
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_str = text[start:end+1]
            return json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        return None
    return None

def validate_and_convert(data: dict) -> Requirements | None:
    """Преобразует словарь в Requirements, обрабатывая возможные расхождения в ключах."""
    try:
        # Нормализуем ключи (на случай, если модель использовала другие названия)
        mapping = {
            "goal": "goal",
            "target_audience": "audience",
            "audience": "audience",
            "features": "features",
            "special_requirements": "special_requirements",
            "constraints": "special_requirements",
        }
        normalized = {}
        for key, value in data.items():
            if key in mapping:
                normalized[mapping[key]] = value
        return Requirements(**normalized)
    except (ValidationError, KeyError, TypeError):
        return None

def save_requirements_to_file(req: Requirements):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"requirements_{timestamp}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(req.model_dump(), f, ensure_ascii=False, indent=2)
    print(f"\nТребования сохранены в файл: {filename}")

def main():
    print("Привет! Опишите идею вашего приложения простыми словами.")
    chat_history = []

    while True:
        user_input = input("\nВы: ").strip()
        if user_input.lower() in ["выход", "exit", "quit"]:
            print("До свидания!")
            break

        chat_history.append(HumanMessage(content=user_input))

        # === Попытка 1: structured output ===
        try:
            response_obj = (prompt | llm.with_structured_output(Requirements)).invoke({"chat_history": chat_history})
            if isinstance(response_obj, Requirements):
                req = response_obj
                success = True
            else:
                success = False
        except Exception:
            success = False

        if not success:
            # === Попытка 2: извлечь JSON из текста ===
            raw_response = chat_chain.invoke({"chat_history": chat_history})
            answer = raw_response.content.strip()
            chat_history.append(AIMessage(content=answer))

            json_data = extract_json_from_text(answer)
            if json_data:
                req = validate_and_convert(json_data)
                if req:
                    success = True
                else:
                    print(f"\nАгент: {answer}")
                    continue
            else:
                print(f"\nАгент: {answer}")
                continue

        # === Успешно получили требования ===
        print("\nФормализованные требования:")
        print(f"- Цель: {req.goal}")
        print("- Основные функции:")
        for i, feat in enumerate(req.features, 1):
            print(f"  {i}. {feat}")
        print(f"- Аудитория: {req.audience}")
        print(f"- Особые требования: {req.special_requirements}")

        # === Меню действий ===
        while True:
            print("\nЧто дальше?")
            print("1. Сохранить в файл")
            print("2. Начать заново")
            print("3. Выйти")
            choice = input("Выберите опцию (1/2/3): ").strip()

            if choice == "1":
                save_requirements_to_file(req)
            elif choice == "2":
                chat_history = []
                print("\n🔄 Начинаем новую сессию!")
                break
            elif choice == "3":
                print("До свидания!")
                return
            else:
                print("Неверный выбор. Попробуйте снова.")

if __name__ == "__main__":
    main()