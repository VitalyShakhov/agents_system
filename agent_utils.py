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

def list_projects() -> list[str]:
    """Возвращает список папок проектов."""
    projects_dir = "projects"
    if not os.path.exists(projects_dir):
        return []
    
    return sorted(
        [os.path.join(projects_dir, d) for d in os.listdir(projects_dir) 
         if os.path.isdir(os.path.join(projects_dir, d))],
        reverse=True
    )

def get_project_path(projects: list[str]) -> str:
    for i, project_path in enumerate(projects, 1):
        metadata_file = os.path.join(project_path, "metadata.json")
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                name = metadata.get("project_name", os.path.basename(project_path))
                created = metadata.get("created_at", "")[:16]
                print(f"{i}. [{created}] {name}")
        except:
            print(f"{i}. {os.path.basename(project_path)}")
    
    print(f"\n{i+1}. Ввести путь вручную")
    
    try:
        choice = int(input(f"\nВыберите проект (1-{len(projects)}): "))
        if 1 <= choice <= len(projects):
            project_path = projects[choice - 1]
        else:
            project_path = input("Введите путь к папке проекта: ").strip()
    except (ValueError, IndexError):
        project_path = input("Введите путь к папке проекта: ").strip()
    
    return project_path

