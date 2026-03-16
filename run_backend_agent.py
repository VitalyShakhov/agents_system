
#!/usr/bin/env python3
"""
Запуск агента бэкенд-разработчика
"""
import sys
import logging
from agents.backend_agent import BackendDeveloperAgent

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backend_agent.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def main():
    if len(sys.argv) < 2:
        print("Использование: python run_backend_agent.py <путь_к_проекту>")
        sys.exit(1)
    
    project_path = sys.argv[1]
    
    print(f"🚀 Запуск бэкенд-агента для проекта: {project_path}")
    
    agent = BackendDeveloperAgent(project_path)
    
    # Загружаем задачи
    tasks = agent.get_assigned_tasks()
    print(f"📋 Найдено задач: {len(tasks)}")
    
    if not tasks:
        print("⚠️  Нет задач для бэкенд-разработчика")
        return
    
    # Генерируем базовую структуру
    agent.generate_project_structure()
    
    # Выполняем задачи
    for task in tasks:
        print(f"\n🔧 Задача {task.id}: {task.title}")
        print(f"   Приоритет: {task.priority}, Оценка: {task.estimated_hours}ч")
        result = agent.execute_task(task)
        status_icon = "✅" if result["status"] == "completed" else "❌"
        print(f"   {status_icon} Статус: {result['status']}")
        if result["output_files"]:
            print(f"   📁 Файлы: {', '.join(result['output_files'][:3])}")
    
    # Сохраняем отчёт
    agent.save_results()
    print(f"\n✨ Готово! Результаты сохранены в: {project_path}/backend/")

if __name__ == "__main__":
    main()