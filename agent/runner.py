import json
import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent

from agent.client import get_client
from agent.prompt import EXECUTOR_PROMPT, PLANNER_PROMPT, REPORTER_PROMPT
from compression import compress_dataframe
from executor import execute_python
from plots import fallback_plots
from utils.security import sanitize_instruction

_current_df: pd.DataFrame | None = None


def _run_python(code: str) -> str:
    result = execute_python(code, _current_df)
    return json.dumps(
        {
            "text_output": result.get("text_output", "")[:3000],
            "error": result.get("error"),
            "images_count": len(result.get("images", [])),
        },
        ensure_ascii=False,
    )


run_python_tool = StructuredTool.from_function(
    func=_run_python,
    name="run_python",
    description="Runs Python code to analyze the dataframe df",
)


def plan_tasks(compressed: dict, instruction: str) -> list[str]:
    llm = get_client()
    prompt = f"""Датасет:
    {json.dumps(compressed, ensure_ascii=False, indent=2)}
    Дополнительные пожелания пользователя: {instruction or 'нет'}
    Придумай список аналитических задач для этого датасета"""

    response = llm.invoke(
        [
            SystemMessage(content=PLANNER_PROMPT),
            HumanMessage(content=prompt),
        ]
    )

    text = response.content.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]

    try:
        tasks = json.loads(text)
        return tasks if isinstance(tasks, list) else []
    except Exception:
        return [
            "Общая статистика и типы данных",
            "Пропущенные значения и качество данных",
            "Распределения числовых колонок",
            "Корреляции между переменными",
            "Выбросы в ключевых колонках",
        ]


def execute_task(task: str, compressed: dict) -> tuple[str, list]:
    llm = get_client()
    agent = create_react_agent(model=llm, tools=[run_python_tool])

    prompt = f"""Задача: {task}

    Датасет:
    {json.dumps(compressed, ensure_ascii=False, indent=2)}"""

    result = agent.invoke(
        {
            "messages": [
                SystemMessage(content=EXECUTOR_PROMPT),
                HumanMessage(content=prompt),
            ]
        },
        config={"recursion_limit": 25},
    )

    tool_output = ""
    images = []
    for msg in result["messages"]:
        if (
            isinstance(msg, AIMessage)
            and hasattr(msg, "tool_calls")
            and msg.tool_calls
        ):
            for tc in msg.tool_calls:
                if tc["name"] == "run_python":
                    exec_result = execute_python(
                        tc["args"].get("code", ""), _current_df
                    )
                    images.extend(exec_result.get("images", []))
                    tool_output += exec_result.get("text_output", "")

    return tool_output, images


def generate_report(tasks_results: list[dict], compressed: dict) -> str:
    llm = get_client()

    results_text = ""
    for i, item in enumerate(tasks_results, 1):
        results_text += (
            f"\n--- Задача {i}: {item['task']} ---\n{item['output']}\n"
        )

    prompt = f"""Датасет (краткая информация):
    {json.dumps(compressed, ensure_ascii=False, indent=2)}

    Результаты анализа по задачам:
    {results_text}

    Напиши финальный отчёт"""

    response = llm.invoke(
        [
            SystemMessage(content=REPORTER_PROMPT),
            HumanMessage(content=prompt),
        ]
    )
    return response.content


def run_agent(df: pd.DataFrame, user_instruction: str = "", model: str = None):
    global _current_df
    _current_df = df.copy()

    compressed = compress_dataframe(df)
    instruction = sanitize_instruction(user_instruction or "")

    tasks = plan_tasks(compressed, instruction)

    all_images = []
    tasks_results = []
    debug_lines = [f"Задачи: {json.dumps(tasks, ensure_ascii=False)}"]

    for i, task in enumerate(tasks, 1):
        output, images = execute_task(task, compressed)
        all_images.extend(images)
        tasks_results.append({"task": task, "output": output})
        debug_lines.append(f"\n[Задача {i}: {task}]\n{output}")

    report = generate_report(tasks_results, compressed)

    if not all_images:
        all_images = fallback_plots(df)

    return report, all_images, "\n".join(debug_lines)
