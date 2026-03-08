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
from agent_utils import extract_json_from_text, get_llm, list_projects, get_project_path

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
            if len(tasks) > 5:
                print("      ...")

def generate_tasks_by_role(requirements: Requirements, tech_stack: TechStack) -> dict:
    """Генерирует задачи для каждой роли отдельно."""
    
    all_tasks = []
    milestones = []
    
    # Определяем контекст для каждой роли
    role_contexts = {
        AgentRole.BACKEND_DEVELOPER: "Серверная логика, API, работа с базой данных, бизнес-логика",
        AgentRole.FRONTEND_DEVELOPER: "Пользовательский интерфейс, визуализация, взаимодействие с пользователем",
        AgentRole.TESTER: "Тестирование функционала, поиск багов, проверка критериев приемки",
        AgentRole.DEVOPS: "Развёртывание, CI/CD, настройка серверов, мониторинг",
        AgentRole.DATABASE_ADMIN: "Проектирование базы данных, оптимизация запросов, миграции",
        AgentRole.SECURITY_SPECIALIST: "Проверка безопасности, защита от уязвимостей, аутентификация",
        # AgentRole.FULLSTACK_DEVELOPER: "Комплексная разработка фронтенда и бэкенда",
    }
    
    # Форматируем данные для промпта
    requirements_json = json.dumps(requirements.model_dump(), ensure_ascii=False, indent=2)
    tech_stack_json = json.dumps(tech_stack.model_dump(), ensure_ascii=False, indent=2)

    # Генерируем задачи для каждой роли
    for role, context in role_contexts.items():
        print(f"\n🔄 Генерация задач для {role.value}...")
      
        role_prefix = role.value.upper().replace('_', '-')
        role_prompt_str = f"""
Ты — технический руководитель. Создай 0-10 конкретных задач для роли: {role.value}
Задач для роли может не быть, это нормально. Задач необязательно должно быть максимальное количество.

Контекст: {context}

Правила:
- Каждая задача должна быть выполнима за 1-8 часов
- Укажи зависимости от других задач (если есть)
- Добавь критерии приемки (2-3 пункта)
- Используй префикс {role_prefix} для ID задач (например: {role_prefix}-001)
- Укажи реалистичную оценку времени в часах

Формат ответа — ТОЛЬКО ЧИСТЫЙ JSON:
{{{{"tasks": [
  {{{{
    "id": "{role_prefix}-001",
    "title": "Краткое название задачи",
    "description": "Подробное описание задачи",
    "role": "{role.value}",
    "priority": 1,
    "estimated_hours": 4,
    "dependencies": [],
    "acceptance_criteria": ["Критерий 1", "Критерий 2"],
    "tags": ["тег1", "тег2"]
  }}}}
]}}}}
""" 

        role_prompt = ChatPromptTemplate.from_messages([
    ("system", role_prompt_str),
    ("human", """
Требования к приложению:
{requirements_json}

Технологический стек:
{tech_stack_json}
"""),
])
        
        try:
            # Отправляем запрос
            response = (role_prompt | llm).invoke({"requirements_json": requirements_json, "tech_stack_json": tech_stack_json})
            
            # Извлекаем JSON
            json_data = extract_json_from_text(response.content)
            
            if json_data and ("tasks" in json_data):
                role_tasks = json_data["tasks"]
                
                # Добавляем задачи в общий список
                for task in role_tasks:
                    all_tasks.append(task)
                
                print(f"✅ Сгенерировано {len(role_tasks)} задач")
            else:
                print(f"⚠️ Не удалось сгенерировать задачи для {role.value}")
                
        except Exception as e:
            print(f"❌ Ошибка для {role.value}: {e}")
            continue
    
    # Формируем итоговый результат
    result = {
        "tasks": all_tasks,
        "milestones": milestones
    }

    return result

def main():
    print("👨‍💼 Агент-тимлид: Планирование проекта")
    print("=" * 50)
    
    # === Шаг 1: Выбор проекта ===
    print("\n📂 Доступные проекты:")
    projects = list_projects()
    if not projects:
        print("❌ Нет сохранённых проектов. Сначала запустите requirements_agent.py и stack_agent.py")
        return
    
    project_path = get_project_path(projects)
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
    
    try:
        plan_dict = generate_tasks_by_role(requirements, tech_stack)         
    except Exception as e:
        print(f"❌ Ошибка при генерации задач: {e}")
        return
        
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