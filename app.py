"""Основной скрипт для запуска и оценки агента"""
import os
import gr
import requests
import pandas as pd
from langchain_core.messages import HumanMessage
from agents.agent import build_graph

# Константы
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

class BasicAgent:
    """Класс агента, построенного на langgraph."""
    def __init__(self):
        print("Инициализация BasicAgent...")
        # Строим граф агента при инициализации
        self.graph = build_graph()

    def __call__(self, question: str) -> str:
        """Основной метод для обработки вопросов.
        
        Args:
            question (str): Вопрос для обработки агентом
            
        Returns:
            str: Ответ агента
        """
        print(f"Агент получил вопрос (первые 50 символов): {question[:50]}...")
        # Создаем сообщение в формате LangChain
        messages = [HumanMessage(content=question)]
        # Запускаем граф агента
        messages = self.graph.invoke({"messages": messages})
        # Извлекаем последний ответ
        answer = messages['messages'][-1].content
        return answer[14:]  # Обрезаем префикс (если есть)


def run_and_submit_all(profile: gr.OAuthProfile | None):
    """Основная функция для запуска агента и отправки ответов.
    
    Args:
        profile: Профиль пользователя Hugging Face (если авторизован)
        
    Returns:
        tuple: Статус выполнения и DataFrame с результатами
    """
    # Получаем ID пространства (для формирования ссылки на код)
    space_id = os.getenv("SPACE_ID")
    
    # Проверяем авторизацию пользователя
    if profile:
        username = f"{profile.username}"
        print(f"Пользователь авторизован: {username}")
    else:
        print("Пользователь не авторизован.")
        return "Пожалуйста, авторизуйтесь в Hugging Face.", None

    # Формируем URL для запросов
    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 1. Инициализация агента
    try:
        agent = BasicAgent()
    except Exception as e:
        print(f"Ошибка при создании агента: {e}")
        return f"Ошибка инициализации агента: {e}", None
    
    # Ссылка на код агента (для проверки)
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(f"Ссылка на код агента: {agent_code}")

    # 2. Получение вопросов от сервера
    print(f"Запрашиваем вопросы с: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        
        if not questions_data:
            print("Получен пустой список вопросов.")
            return "Список вопросов пуст или имеет неверный формат.", None
            
        print(f"Получено {len(questions_data)} вопросов.")
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при получении вопросов: {e}")
        return f"Ошибка получения вопросов: {e}", None
    except Exception as e:
        print(f"Неожиданная ошибка при получении вопросов: {e}")
        return f"Неожиданная ошибка: {e}", None

    # 3. Обработка вопросов агентом
    results_log = []  # Для хранения логов
    answers_payload = []  # Для отправки на сервер
    
    print(f"Запускаем агента на {len(questions_data)} вопросах...")
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        
        if not task_id or question_text is None:
            print(f"Пропускаем вопрос с отсутствующим ID или текстом: {item}")
            continue
            
        try:
            # Получаем ответ от агента
            submitted_answer = agent(question_text)
            answers_payload.append({
                "task_id": task_id,
                "submitted_answer": submitted_answer
            })
            results_log.append({
                "Task ID": task_id,
                "Question": question_text,
                "Submitted Answer": submitted_answer
            })
        except Exception as e:
            print(f"Ошибка при обработке вопроса {task_id}: {e}")
            results_log.append({
                "Task ID": task_id,
                "Question": question_text,
                "Submitted Answer": f"ОШИБКА АГЕНТА: {e}"
            })

    if not answers_payload:
        print("Агент не сгенерировал ни одного ответа.")
        return "Агент не сгенерировал ответов.", pd.DataFrame(results_log)

    # 4. Подготовка данных для отправки
    submission_data = {
        "username": username.strip(),
        "agent_code": agent_code,
        "answers": answers_payload
    }
    
    status_update = f"Агент завершил работу. Отправляем {len(answers_payload)} ответов для пользователя '{username}'..."
    print(status_update)

    # 5. Отправка ответов на сервер
    print(f"Отправляем ответы на: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        
        # Формируем итоговый статус
        final_status = (
            f"Успешная отправка!\n"
            f"Пользователь: {result_data.get('username')}\n"
            f"Общий счет: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} правильных)\n"
            f"Сообщение: {result_data.get('message', 'Нет сообщения.')}"
        )
        
        print("Отправка успешно завершена.")
        results_df = pd.DataFrame(results_log)
        return final_status, results_df
        
    except requests.exceptions.RequestException as e:
        error_msg = f"Ошибка при отправке: {e}"
        print(error_msg)
        results_df = pd.DataFrame(results_log)
        return error_msg, results_df
    except Exception as e:
        error_msg = f"Неожиданная ошибка: {e}"
        print(error_msg)
        results_df = pd.DataFrame(results_log)
        return error_msg, results_df


# Создаем интерфейс Gradio
with gr.Blocks() as demo:
    gr.Markdown("# Система оценки агента")
    gr.Markdown(
        """
        **Инструкции:**
        1. Клонируйте это пространство и модифицируйте код агента
        2. Авторизуйтесь через Hugging Face
        3. Нажмите кнопку для запуска оценки
        """
    )

    # Элементы интерфейса
    gr.LoginButton()
    run_button = gr.Button("Запустить оценку и отправить ответы")
    
    # Выходные элементы
    status_output = gr.Textbox(label="Статус выполнения", lines=5, interactive=False)
    results_table = gr.DataFrame(label="Вопросы и ответы агента", wrap=True)

    # Привязка кнопки к функции
    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table]
    )


if __name__ == "__main__":
    print("\n" + "-"*30 + " Запуск приложения " + "-"*30)
    
    # Проверяем переменные окружения
    space_host = os.getenv("SPACE_HOST")
    space_id = os.getenv("SPACE_ID")
    
    if space_host:
        print(f"SPACE_HOST: {space_host}")
        print(f"URL: https://{space_host}.hf.space")
    
    if space_id:
        print(f"SPACE_ID: {space_id}")
        print(f"Репозиторий: https://huggingface.co/spaces/{space_id}")
    
    print("-"*75 + "\n")
    print("Запускаем интерфейс Gradio...")
    demo.launch(debug=True, share=False)