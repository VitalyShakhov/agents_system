import os
import re
import json
from pathlib import Path
from typing import Dict, List, Any
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from .base_agent import AgentInterface, Task

load_dotenv()

class BackendDeveloperAgent(AgentInterface):
    """Агент для генерации бэкенд-кода"""
    
    def __init__(self, project_path: str):
        super().__init__(project_path, "backend_developer")
        
        # Инициализация LLM
        # todo get_llm
        self.llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
            temperature=0.2  # Низкая температура для точного кода
        )
        
        # Определяем выходную директорию для кода
        self.output_dir = Path(project_path) / "backend"
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "src").mkdir(exist_ok=True)
        (self.output_dir / "tests").mkdir(exist_ok=True)
    
    def _get_tech_context(self) -> str:
        """Формирует контекст технологий для промпта"""
        stack = self.context.get("tech_stack", {})
        return f"""
Язык программирования: {stack.get('language', 'Python')}
Фреймворк: {stack.get('framework', 'FastAPI')}
База данных: {stack.get('database', 'SQLite')}
Архитектура: {stack.get('architecture', 'Монолит')}
Дополнительные инструменты: {', '.join(stack.get('additional_tools', []))}
"""
    
    def _code_generation_prompt(self, task: Task) -> ChatPromptTemplate:
        """Создаёт промпт для генерации кода"""
        # tech_context = self._get_tech_context()
        # requirements = json.dumps(self.context.get("requirements", {}), ensure_ascii=False, indent=2)
    
        # Системный промпт содержит ТОЛЬКО шаблон с переменными {имя}
        system_prompt_template = """
Ты — опытный бэкенд-разработчик. Твоя задача — написать рабочий, готовый к использованию код.

Требования к приложению:
{requirements_json}

Технологический стек:
{tech_stack_json}

ПРАВИЛА ГЕНЕРАЦИИ:
1. Пиши ПОЛНОСТЬЮ рабочий код без заглушек (никаких "TODO", "implement me")
2. Используй правильные импорты и структуру проекта для выбранного стека
3. Возвращай ТОЛЬКО код без пояснений, обёрнутый в ```язык ... ```
"""
    
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt_template),
            ("human", "ЗАДАЧА:\nНазвание: {task_title}\n\nОписание: {task_description}\n\nКритерии приёмки:\n{acceptance_criteria}")
        ])
    
    def _extract_code_from_response(self, response: str) -> Dict[str, str]:
        """
        Извлекает код из ответа модели.
        Возвращает словарь: {расширение_файла: код}
        """
        # Ищем блоки кода в формате ```язык ... ```
        code_blocks = re.findall(r'```(\w+)?\s*(.*?)```', response, re.DOTALL)
        
        if not code_blocks:
            # Если нет блоков — возвращаем весь ответ как код
            return {"py": response.strip()}
        
        results = {}
        for lang, code in code_blocks:
            lang = lang.lower() if lang else "py"
            # Определяем расширение файла
            ext_map = {
                "python": "py", "py": "py",
                "javascript": "js", "js": "js",
                "typescript": "ts", "ts": "ts",
                ".NET": "cs", "C#": "cs", "csharp": "cs",
                "sql": "sql",
                "json": "json", "xml": "xml",
                "yaml": "yaml", "yml": "yml"
            }
            ext = ext_map.get(lang, "py")
            results[ext] = code.strip()
        
        return results

    def execute_task(self, task: Task) -> Dict[str, Any]:
        """Выполняет задачу — генерирует код"""
        self.logger.info(f"Начинаю выполнение задачи {task.id}: {task.title}")
        
        try:
            # Генерируем код
            prompt = self._code_generation_prompt(task)
            response = (prompt | self.llm).invoke({
                "requirements_json": json.dumps(self.context.get("requirements", {}), ensure_ascii=False, indent=2),
                "tech_stack_json": json.dumps(self.context.get("tech_stack", {}), ensure_ascii=False, indent=2),
                # "stack_instructions": self._get_stack_instructions(),  # метод с текстовыми инструкциями под стек
                "task_title": task.title,
                "task_description": task.description,
                "acceptance_criteria": "\n".join(f"- {c}" for c in task.acceptance_criteria)
            })
            raw_code = response.content
            
            self.logger.debug(f"Получен код для {task.id} (длина: {len(raw_code)} символов)")
            
            # Извлекаем код из ответа
            code_files = self._extract_code_from_response(raw_code)
            
            # Сохраняем файлы
            saved_files = []
            for ext, code in code_files.items():
                # Определяем имя файла на основе задачи
                filename = f"{task.id.lower().replace('-', '_')}.{ext}"
                filepath = self.output_dir / "src" / filename
                
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(code)
                
                saved_files.append(str(filepath.relative_to(self.project_path)))
                self.logger.info(f"Сохранён файл: {filepath.name}")
            
            result = {
                "task_id": task.id,
                "status": "completed",
                "output_files": saved_files,
                "code_snippets": [{"language": ext, "code": code[:200] + "..."} for ext, code in code_files.items()],
                "notes": f"Сгенерирован код для задачи '{task.title}'",
                "errors": []
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка при выполнении задачи {task.id}: {e}")
            result = {
                "task_id": task.id,
                "status": "failed",
                "output_files": [],
                "code_snippets": [],
                "notes": f"Ошибка: {str(e)}",
                "errors": [str(e)]
            }
        
        # Сохраняем результат задачи
        self.results[task.id] = result
        return result

    def generate_project_structure(self):
        """Генерирует базовую структуру проекта (файлы инициализации, конфиги)"""
        stack = self.context.get("tech_stack", {})
        language = stack.get("language", "Python").lower()
        
        if "python" in language:
            # requirements.txt
            reqs = self._generate_requirements(stack)
            with open(self.output_dir / "requirements.txt", "w", encoding="utf-8") as f:
                f.write(reqs)
            
            # .gitignore
            with open(self.output_dir / ".gitignore", "w", encoding="utf-8") as f:
                f.write("*.pyc\n__pycache__/\n.env\nvenv/\n")
            
            self.logger.info("Создана базовая структура backend-проекта")

    # todo проект может быть не только на python
    def _generate_requirements(self, stack: Dict) -> str:
        """Генерирует содержимое requirements.txt на основе стека"""
        base_reqs = ["python-dotenv", "pydantic>=2.0.0"]
        
        framework = stack.get("framework", "").lower()
        if "fastapi" in framework:
            base_reqs.extend(["fastapi", "uvicorn"])
        elif "django" in framework:
            base_reqs.append("Django")
        elif "flask" in framework:
            base_reqs.append("Flask")
        
        db = stack.get("database", "").lower()
        if "postgresql" in db:
            base_reqs.append("psycopg2-binary")
        elif "sqlite" in db:
            base_reqs.append("sqlite3")  # встроен в Python
        elif "mongodb" in db:
            base_reqs.append("pymongo")
        
        return "\n".join(base_reqs) + "\n"
    
