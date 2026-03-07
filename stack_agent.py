import os
import json
import re
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field, ValidationError
from langchain_openai import ChatOpenAI
from models import Requirements, TechStackRecommendation
from prompts import clarification_prompt_str, stack_selection_prompt_str
from agent_utils import extract_json_from_text, get_llm

load_dotenv()

# === Модель уточняющих вопросов ===
class ClarificationNeeded(BaseModel):
    needs_clarification: bool = Field(description="Требуются ли уточнения")
    questions: list[str] = Field(description="Список уточняющих вопросов")

# === Инициализация модели ===
llm = get_llm(0.4)

# === Промпт для анализа требований и определения, нужны ли уточнения ===
clarification_prompt = ChatPromptTemplate.from_messages([
    ("system", clarification_prompt_str),
    ("human", "Требования к приложению:\n{requirements_json}"),
])

# === Промпт для подбора стека ===
stack_selection_prompt = ChatPromptTemplate.from_messages([
    ("system", stack_selection_prompt_str),
    ("human", "Требования к приложению:\n{requirements_json}\n\nДополнительная информация:\n{additional_info}"),
])

def load_requirements_from_project(project_path: str) -> tuple[Requirements, dict] | None:
    """Загружает требования из папки проекта."""
    try:
        # Загрузить метаданные
        metadata_file = os.path.join(project_path, "metadata.json")
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        # Загрузить требования
        requirements_file = os.path.join(project_path, "requirements.json")
        with open(requirements_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        requirements = Requirements(**data)
        return requirements, metadata
    except (Exception) as e:
        print(f"❌ Ошибка при загрузке: {e}")
        return None

def save_recommendation_to_project(project_path: str, rec: TechStackRecommendation, requirements: Requirements):
    """Сохраняет рекомендации в папку проекта."""
    # Сохранить рекомендации
    stack_file = os.path.join(project_path, "tech_stack.json")
    output = {
        "requirements": requirements.model_dump(),
        "recommendation": rec.model_dump()
    }
    with open(stack_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    # Обновить метаданные
    metadata_file = os.path.join(project_path, "metadata.json")
    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    metadata["stack_generated_at"] = datetime.now().isoformat()
    
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"\n📁 Рекомендации сохранены в: {stack_file}")

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

def main():
    print("🛠️  Агент подбора технологического стека")
    print("=" * 50)
    
    # === Шаг 1: Выбор проекта ===
    print("\n📂 Доступные проекты:")
    projects = list_projects()
    
    if not projects:
        print("❌ Нет сохранённых проектов. Сначала запустите requirements_agent.py")
        return
    
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
    
    if not os.path.exists(project_path):
        print(f"❌ Папка не найдена: {project_path}")
        return
    
    # === Шаг 2: Загрузка требований ===
    print(f"\nЗагрузка требований из: {project_path}")
    result = load_requirements_from_project(project_path)
    if not result:
        return
    
    requirements, metadata = result
    
    print("\n✅ Требования загружены:")
    print(f"- Цель: {requirements.goal}")
    print("- Функции:")
    for feat in requirements.features:
        print(f"  • {feat}")
    print(f"- Аудитория: {requirements.audience}")
    print(f"- Особые требования: {requirements.special_requirements}")
    
    # === Шаг 3: Определение, нужны ли уточнения ===
    print("\n🔍 Анализ требований...")
    
    requirements_json = json.dumps(requirements.model_dump(), ensure_ascii=False, indent=2)
    
    try:
        # Попытка получить структурированный ответ
        clarification_chain = clarification_prompt | llm.with_structured_output(ClarificationNeeded)
        clarification_result = clarification_chain.invoke({"requirements_json": requirements_json})
        needs_clarification = clarification_result.needs_clarification
        questions = clarification_result.questions
    except Exception as e:
        # Fallback: извлечение из текста
        print(f"⚠️ Fallback к текстовому анализу: {e}")
        raw_response = (clarification_prompt | llm).invoke({"requirements_json": requirements_json})
        json_data = extract_json_from_text(raw_response.content)
        
        if json_data:
            needs_clarification = json_data.get("needs_clarification", False)
            questions = json_data.get("questions", [])
        else:
            needs_clarification = False
            questions = []
    
    additional_info = ""
    
    if needs_clarification and questions:
        print("\n📝 Требуются уточнения для подбора оптимального стека:")
        for i, question in enumerate(questions, 1):
            print(f"{i}. {question}")
        
        print("\nПожалуйста, ответьте на вопросы:")
        answers = []
        for q in questions:
            answer = input(f"→ {q}\nОтвет: ").strip()
            answers.append(f"Вопрос: {q}\nОтвет: {answer}")
        
        additional_info = "\n\n".join(answers)
        print("\n✅ Уточнения получены.")
    else:
        print("✅ Уточнения не требуются. Переходим к подбору стека...")
        additional_info = "Не требуется"
    
    # === Шаг 4: Подбор стека ===
    print("\n⚙️  Подбираю оптимальный технологический стек...")
    
    try:
        stack_chain = stack_selection_prompt | llm.with_structured_output(TechStackRecommendation)
        recommendation = stack_chain.invoke({
            "requirements_json": requirements_json,
            "additional_info": additional_info
        })
        
        # if not isinstance(recommendation, TechStackRecommendation):
        #     raise ValueError("Некорректный формат рекомендации")
    except Exception as e:
        print(f"⚠️ Fallback к текстовому выводу: {e}")
        raw_response = (stack_selection_prompt | llm).invoke({
            "requirements_json": requirements_json,
            "additional_info": additional_info
        })
        
        json_data = extract_json_from_text(raw_response.content)
        if not json_data:
            print("❌ Не удалось извлечь рекомендации. Попробуйте снова.")
            return
        
        try:
            recommendation = TechStackRecommendation(**json_data)
        except ValidationError as ve:
            print(f"❌ Ошибка валидации: {ve}")
            return
    
    # === Шаг 5: Вывод и сохранение ===
    print("\n" + "=" * 50)
    print("✅ Рекомендации по технологическому стеку")
    print("=" * 50)
    
    print(f"\n🔹 Язык программирования: {recommendation.language}")
    print(f"🔹 Фреймворк: {recommendation.framework}")
    print(f"🔹 База данных: {recommendation.database}")
    if recommendation.frontend:
        print(f"🔹 Фронтенд: {recommendation.frontend}")
    print(f"🔹 Архитектура: {recommendation.architecture}")
    print(f"🔹 Хостинг/Деплой: {recommendation.hosting}")
    
    print(f"\n🔹 Дополнительные инструменты:")
    for tool in recommendation.additional_tools:
        print(f"   • {tool}")
    
    print(f"\n🔹 Обоснование:")
    print(f"   {recommendation.justification}")
    
    print(f"\n🔹 Заметки о масштабируемости:")
    print(f"   {recommendation.scalability_notes}")
    
    # Сохранить в папку проекта
    save_recommendation_to_project(project_path, recommendation, requirements)
    
    print("\n✨ Готово! Рекомендации добавлены в проект.")
    print(f"📁 Папка проекта: {project_path}")

if __name__ == "__main__":
    main()