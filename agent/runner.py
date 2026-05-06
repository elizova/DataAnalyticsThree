from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent
from agent.client import get_client
from agent.prompt import SYSTEM_PROMPT
from compression import compress_dataframe
from executor import execute_python
from plots import fallback_plots
from utils.security import sanitize_instruction
import json
import pandas as pd

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


def run_agent(df: pd.DataFrame, user_instruction: str = "", model: str = None):
    global _current_df
    _current_df = df.copy()

    compressed = compress_dataframe(df)
    instruction = sanitize_instruction(user_instruction or "")

    agent = create_react_agent(
        model=get_client(),
        tools=[run_python_tool],
    )

    dataset_info = f"""Датасет:
    {json.dumps(compressed, ensure_ascii=False, indent=2)}

    Инструкция пользователя:
    {instruction}

    сначала вызови инструмент run_python с кодом анализа, затем напиши отчёт."""

    result = agent.invoke(
        {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=dataset_info),
            ]
        },
        config={"recursion_limit": 15},
    )

    for i, msg in enumerate(result["messages"]):
        print(f"[{i}] {type(msg).__name__}: {repr(str(msg.content)[:120])}")

    report = ""
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content and msg.content.strip():
            report = msg.content
            break

    all_images = []
    debug_lines = []
    for msg in result["messages"]:
        if (
            isinstance(msg, AIMessage)
            and hasattr(msg, "tool_calls")
            and msg.tool_calls
        ):
            for tc in msg.tool_calls:
                if tc["name"] == "run_python":
                    code = tc["args"].get("code", "")
                    exec_result = execute_python(code, _current_df)
                    all_images.extend(exec_result.get("images", []))
                    debug_lines.append(
                        f"[run_python]\n{exec_result.get('text_output', '')}\n"
                    )

    if not all_images:
        all_images = fallback_plots(df)

    return report, all_images, "\n".join(debug_lines)
