from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any
from pydantic import BaseModel, Field
import json
import logging

# Модель задачи (должна совпадать с той, что использует тимлид)
class Task(BaseModel):
    id: str
    title: str
    description: str
    role: str
    priority: int
    estimated_hours: int
    dependencies: List[str]
    acceptance_criteria: List[str]
    tags: List[str]

class AgentInterface(ABC):
    """Базовый интерфейс для всех агентов-разработчиков"""
    
    def __init__(self, project_path: str, role_name: str):
        self.project_path = Path(project_path)
        self.role_name = role_name
        self.logger = logging.getLogger(f"agent.{role_name}")
        self.context = self._load_context()
        self.assigned_tasks: List[Task] = []
        self.results: Dict[str, Any] = {}
    
    def _load_context(self) -> Dict[str, Any]:
        """Загружает контекст проекта: требования, стек, план"""
        context = {}
        
        # Требования
        req_file = self.project_path / "requirements.json"
        if req_file.exists():
            with open(req_file, "r", encoding="utf-8") as f:
                context["requirements"] = json.load(f)
        
        # Технологический стек
        stack_file = self.project_path / "tech_stack.json"
        if stack_file.exists():
            with open(stack_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                context["tech_stack"] = data.get("recommendation", data)
        
        # План работ
        plan_file = self.project_path / "project_plan.json"
        if plan_file.exists():
            with open(plan_file, "r", encoding="utf-8") as f:
                context["project_plan"] = json.load(f)
        
        return context
    
    def get_assigned_tasks(self) -> List[Task]:
        """Получает задачи, назначенные этой роли"""
        if "project_plan" not in self.context:
            return []
        
        all_tasks = self.context["project_plan"].get("tasks", [])
        self.assigned_tasks = [
            Task(**task) for task in all_tasks 
            if task.get("role") == self.role_name
        ]
        return self.assigned_tasks
    
    @abstractmethod
    def execute_task(self, task: Task) -> Dict[str, Any]:
        """
        Выполняет задачу и возвращает результат.
        
        Возвращает:
            {
                "task_id": str,
                "status": "completed" | "failed" | "partial",
                "output_files": List[str],  # пути к созданным файлам
                "code_snippets": List[Dict],  # сгенерированный код
                "notes": str,  # комментарии агента
                "errors": List[str]  # ошибки (если есть)
            }
        """
        pass
    
    def save_results(self):
        """Сохраняет результаты работы в папку проекта"""
        output_dir = self.project_path / self.role_name
        output_dir.mkdir(exist_ok=True)
        
        # Сохраняем отчёт
        report_file = output_dir / "implementation_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump({
                "role": self.role_name,
                "executed_at": self._get_timestamp(),
                "tasks_completed": len(self.results),
                "results": self.results
            }, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Результаты сохранены: {report_file}")
    
    def _get_timestamp(self) -> str:
        from datetime import datetime
        return datetime.now().isoformat()