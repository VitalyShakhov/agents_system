import os
import json
import re
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import ValidationError
from langchain_openai import ChatOpenAI
from models import Requirements, print_requirements
from prompts import requirements_prompt_str
from agent_utils import extract_json_from_text, get_llm, sanitize_filename

load_dotenv()

llm = get_llm(0.3)

prompt = ChatPromptTemplate.from_messages([
    ("system", requirements_prompt_str),
    MessagesPlaceholder(variable_name="chat_history"),
])

chat_chain = prompt | llm

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

def save_requirements_to_project(req: Requirements, project_name: str):
    """Сохраняет требования в папку проекта."""
    # Создать имя папки из названия проекта
    folder_name = sanitize_filename(project_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{timestamp}_{folder_name}"
    
    project_path = os.path.join("projects", folder_name)
    os.makedirs(project_path, exist_ok=True)
    
    # Сохранить требования
    requirements_file = os.path.join(project_path, "requirements.json")
    with open(requirements_file, "w", encoding="utf-8") as f:
        json.dump(req.model_dump(), f, ensure_ascii=False, indent=2)
    
    # Создать файл метаданных
    metadata = {
        "project_name": project_name,
        "created_at": datetime.now().isoformat(),
        "folder_name": folder_name
    }
    metadata_file = os.path.join(project_path, "metadata.json")
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"\n📁 Требования сохранены в: {project_path}")
    print(f"   Файл: {requirements_file}")
    return project_path

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
        print_requirements(req)

        # === Сохранение ===
        print("\n💾 Сохранение проекта...")
        project_name = input("Введите название проекта: ").strip()
        if not project_name:
            project_name = req.goal[:50]  # Берём первые 50 символов цели
        
        project_path = save_requirements_to_project(req, project_name)

        # === Меню действий ===
        while True:
            print("\nЧто дальше?")
            print("1. Начать новый проект")
            print("2. Выйти")
            choice = input("Выберите опцию (1/2): ").strip()

            if choice == "1":
                chat_history = []
                print("\n🔄 Начинаем новый проект!")
                break
            elif choice == "2":
                print("До свидания!")
                return
            else:
                print("Неверный выбор. Попробуйте снова.")

if __name__ == "__main__":
    main()