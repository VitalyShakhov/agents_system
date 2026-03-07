import os
import json
from datetime import datetime
from typing import List, Dict, Optional
from enum import Enum
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from models import Requirements, TechStackRecommendation as TechStack
from prompts import task_generation_prompt_str
from agent_utils import extract_json_from_text, get_llm

load_dotenv()

# === Роли агентов ===
class AgentRole(str, Enum):
    BACKEND_DEVELOPER = "backend_developer"
    FRONTEND_DEVELOPER = "frontend_developer"
    TESTER = "tester"
    DEVOPS = "devops"
    DATABASE_ADMIN = "database_admin"
    SECURITY_SPECIALIST = "security_specialist"
    FULLSTACK_DEVELOPER = "fullstack_developer"

# === Модель задачи ===
class Task(BaseModel):
    id: str = Field(description="Уникальный идентификатор задачи (например, 'TASK-001')")
    title: str = Field(description="Краткое название задачи")
    description: str = Field(description="Подробное описание задачи")
    role: AgentRole = Field(description="Роль агента, который должен выполнить задачу")
    priority: int = Field(description="Приоритет (1-5, где 1 - срочно)")
    estimated_hours: int = Field(description="Оценка времени в часах")
    dependencies: List[str] = Field(description="Список ID задач, от которых зависит эта задача")
    acceptance_criteria: List[str] = Field(description="Критерии приемки задачи")
    tags: List[str] = Field(description="Теги для категоризации")

# === Модель плана работ ===
class ProjectPlan(BaseModel):
    project_name: str = Field(description="Название проекта")
    total_tasks: int = Field(description="Общее количество задач")
    tasks_by_role: Dict[str, int] = Field(description="Количество задач по ролям")
    total_estimated_hours: int = Field(description="Общая оценка времени")
    tasks: List[Task] = Field(description="Список всех задач")
    milestones: List[str] = Field(description="Ключевые этапы проекта")

# === Интерфейс агента ===
class AgentInterface:
    """Базовый интерфейс для всех подчинённых агентов"""
    
    def __init__(self, role: AgentRole):
        self.role = role
    
    def get_assigned_tasks(self, project_plan: ProjectPlan) -> List[Task]:
        """Получить задачи, назначенные этой роли"""
        return [task for task in project_plan.tasks if task.role == self.role]
    
    def can_execute_task(self, task: Task) -> bool:
        """Проверить, может ли агент выполнить задачу"""
        return task.role == self.role
    
    def generate_implementation(self, task: Task, context: Dict) -> str:
        """Сгенерировать реализацию задачи (будет переопределено в конкретных агентах)"""
        raise NotImplementedError("Метод должен быть реализован в подклассе")

# === Инициализация модели ===
llm = get_llm(0.5)

# === Промпт для генерации задач ===
task_generation_prompt = ChatPromptTemplate.from_messages([
    ("system", task_generation_prompt_str),
    ("human", """
Требования к приложению:
{requirements_json}

Технологический стек:
{tech_stack_json}
"""),
])

def load_project_data(project_path: str) -> tuple[Requirements, TechStack, dict] | None:
    """Загружает требования и стек из папки проекта."""
    try:
        # Загрузить метаданные
        metadata_file = os.path.join(project_path, "metadata.json")
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        # Загрузить требования
        requirements_file = os.path.join(project_path, "requirements.json")
        with open(requirements_file, "r", encoding="utf-8") as f:
            req_data = json.load(f)
        requirements = Requirements(**req_data)
        
        # Загрузить стек
        stack_file = os.path.join(project_path, "tech_stack.json")
        if not os.path.exists(stack_file):
            print(f"⚠️  Файл tech_stack.json не найден. Запустите сначала stack_agent.py")
            return None
        
        with open(stack_file, "r", encoding="utf-8") as f:
            stack_data = json.load(f)
        
        # tech_stack.json может содержать вложенный объект
        if "recommendation" in stack_data:
            stack_data = stack_data["recommendation"]
        
        tech_stack = TechStack(**stack_data)
        
        return requirements, tech_stack, metadata
    except Exception as e:
        print(f"❌ Ошибка при загрузке: {e}")
        return None

def save_project_plan(project_path: str, plan: ProjectPlan):
    """Сохраняет план работ в папку проекта."""
    plan_file = os.path.join(project_path, "project_plan.json")
    
    with open(plan_file, "w", encoding="utf-8") as f:
        json.dump(plan.model_dump(), f, ensure_ascii=False, indent=2)
    
    print(f"\n📁 План работ сохранён: {plan_file}")
    
    # Обновить метаданные
    metadata_file = os.path.join(project_path, "metadata.json")
    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    metadata["plan_generated_at"] = datetime.now().isoformat()
    metadata["total_tasks"] = plan.total_tasks
    metadata["total_hours"] = plan.total_estimated_hours
    
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

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

def print_task_summary(plan: ProjectPlan):
    """Выводит сводку по задачам."""
    print("\n" + "=" * 60)
    print(f"📋 План работ: {plan.project_name}")
    print("=" * 60)
    print(f"\n📊 Общая статистика:")
    print(f"   • Всего задач: {plan.total_tasks}")
    print(f"   • Общее время: {plan.total_estimated_hours} часов (~{plan.total_estimated_hours/8:.1f} рабочих дней)")
    print(f"   • Ролей задействовано: {len(plan.tasks_by_role)}")
    
    print(f"\n👥 Распределение по ролям:")
    role_names = {
        "backend_developer": "🐍 Backend",
        "frontend_developer": "🎨 Frontend",
        "tester": "🧪 Тестирование",
        "devops": "🚀 DevOps",
        "database_admin": "🗄️  База данных",
        "security_specialist": "🔒 Безопасность",
        "fullstack_developer": "⚡ Fullstack"
    }
    
    for role, count in sorted(plan.tasks_by_role.items(), key=lambda x: x[1], reverse=True):
        role_name = role_names.get(role, role)
        print(f"   • {role_name}: {count} задач")
    
    print(f"\n🎯 Ключевые этапы:")
    for i, milestone in enumerate(plan.milestones, 1):
        print(f"   {i}. {milestone}")
    
    print(f"\n📝 Задачи по приоритетам:")
    for priority in range(1, 6):
        tasks = [t for t in plan.tasks if t.priority == priority]
        if tasks:
            print(f"\n   ⚡ Приоритет {priority} ({len(tasks)} задач):")
            for task in tasks[:5]:  # Показываем первые 5
                print(f"      • {task.id}: {task.title}")

def main():
    print("👨‍💼 Агент-тимлид: Планирование проекта")
    print("=" * 50)
    
    # === Шаг 1: Выбор проекта ===
    print("\n📂 Доступные проекты:")
    projects = list_projects()
    
    if not projects:
        print("❌ Нет сохранённых проектов. Сначала запустите requirements_agent.py и stack_agent.py")
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
    
    # === Шаг 2: Загрузка данных проекта ===
    print(f"\n📂 Загрузка данных проекта...")
    result = load_project_data(project_path)
    if not result:
        return
    
    requirements, tech_stack, metadata = result
    
    print("\n✅ Данные проекта загружены:")
    print(f"- Проект: {metadata.get('project_name', 'Без названия')}")
    print(f"- Технологии: {tech_stack.language} + {tech_stack.framework}")
    if tech_stack.frontend:
        print(f"- Фронтенд: {tech_stack.frontend}")
    print(f"- База данных: {tech_stack.database}")
    
    # === Шаг 3: Генерация задач ===
    print("\n📝 Генерация плана работ...")
    
    requirements_json = json.dumps(requirements.model_dump(), ensure_ascii=False, indent=2)
    tech_stack_json = json.dumps(tech_stack.model_dump(), ensure_ascii=False, indent=2)
    
    available_roles = ", ".join([role.value for role in AgentRole])
    
    try:
        # Попытка структурированного вывода
        chain = task_generation_prompt | llm.with_structured_output(ProjectPlan)
        raw_plan = chain.invoke({
            "requirements_json": requirements_json,
            "tech_stack_json": tech_stack_json,
            "available_roles": available_roles
        })
        
        if isinstance(raw_plan, dict):
            # Если вернулся словарь, преобразуем в ProjectPlan
            plan_dict = raw_plan
        else:
            # Если уже ProjectPlan
            plan_dict = raw_plan.model_dump()
        
    except Exception as e:
        print(f"⚠️ Fallback к текстовому выводу: {e}")
        raw_response = (task_generation_prompt | llm).invoke({
            "requirements_json": requirements_json,
            "tech_stack_json": tech_stack_json,
            "available_roles": available_roles
        })
        
        json_data = extract_json_from_text(raw_response.content)
        if not json_data:
            print("❌ Не удалось извлечь план работ.")
            return
        
        plan_dict = json_data
    
    # Создать план работ
    try:
        # Подсчитать статистику
        tasks_by_role = {}
        total_hours = 0
        
        for task_data in plan_dict.get("tasks", []):
            role = task_data.get("role", "backend_developer")
            tasks_by_role[role] = tasks_by_role.get(role, 0) + 1
            total_hours += task_data.get("estimated_hours", 0)
        
        project_plan = ProjectPlan(
            project_name=metadata.get("project_name", "Новый проект"),
            total_tasks=len(plan_dict.get("tasks", [])),
            tasks_by_role=tasks_by_role,
            total_estimated_hours=total_hours,
            tasks=[Task(**task) for task in plan_dict.get("tasks", [])],
            milestones=plan_dict.get("milestones", [])
        )
    except Exception as e:
        print(f"❌ Ошибка создания плана: {e}")
        return
    
    # === Шаг 4: Вывод и сохранение ===
    print_task_summary(project_plan)
    
    # Сохранить в папку проекта
    save_project_plan(project_path, project_plan)
    
    # === Шаг 5: Экспорт задач по ролям ===
    print("\n" + "=" * 60)
    print("📤 Экспорт задач по ролям")
    print("=" * 60)
    
    export_choice = input("\nХотите экспортировать задачи по отдельным файлам для каждой роли? (да/нет): ").strip().lower()
    
    if export_choice in ["да", "yes", "y", ""]:
        roles_dir = os.path.join(project_path, "tasks_by_role")
        os.makedirs(roles_dir, exist_ok=True)
        
        for role in AgentRole:
            role_tasks = [task for task in project_plan.tasks if task.role == role]
            if role_tasks:
                role_file = os.path.join(roles_dir, f"{role.value}.json")
                with open(role_file, "w", encoding="utf-8") as f:
                    json.dump({
                        "role": role.value,
                        "tasks_count": len(role_tasks),
                        "tasks": [task.model_dump() for task in role_tasks]
                    }, f, ensure_ascii=False, indent=2)
                print(f"   ✅ {role.value}: {len(role_tasks)} задач → {role_file}")
        
        print(f"\n📁 Задачи экспортированы в: {roles_dir}")
    
    print("\n✨ Готово! План работ создан.")
    print(f"📁 Папка проекта: {project_path}")

if __name__ == "__main__":
    main()