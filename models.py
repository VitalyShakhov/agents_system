from pydantic import BaseModel, Field
from typing import Optional

# === Модель требований ===
class Requirements(BaseModel):
    goal: str = Field(description="Цель приложения")
    features: list[str] = Field(description="Список основных функций")
    audience: str = Field(description="Целевая аудитория")
    special_requirements: str = Field(description="Особые условия")

# === Модель рекомендаций по стеку (выход) ===
class TechStackRecommendation(BaseModel):
    language: str = Field(description="Язык программирования")
    framework: str = Field(description="Основной фреймворк/библиотека")
    database: str = Field(description="База данных")
    frontend: Optional[str] = Field(description="Фронтенд-технология (если применимо)")
    architecture: str = Field(description="Архитектурный подход")
    hosting: str = Field(description="Варианты хостинга/деплоя")
    additional_tools: list[str] = Field(description="Дополнительные инструменты (тестирование, сборка и т.д.)")
    justification: str = Field(description="Обоснование выбора стека")
    scalability_notes: str = Field(description="Заметки о масштабируемости и ограничениях")

def print_requirements(requirements: Requirements):
    print(f"- Цель: {requirements.goal}")
    print("- Основные функции:")
    for i, feat in enumerate(requirements.features, 1):
        print(f"  {i}. {feat}")
    print(f"- Аудитория: {requirements.audience}")
    print(f"- Особые требования: {requirements.special_requirements}")    

