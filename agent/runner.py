import json

from agent.client import get_client
from agent.prompt import SYSTEM_PROMPT
from compression import compress_dataframe
from executor import execute_python
from plots import fallback_plots
from utils.security import sanitize_instruction


def run_agent(
    df, user_instruction="", model="meta-llama/llama-3.1-8b-instruct"
):
    compressed = compress_dataframe(df)
    instruction = sanitize_instruction(user_instruction or "")

    dataset_info = f"""Датасет:

    {json.dumps(compressed, ensure_ascii=False, indent=2)}

    Инструкция пользователя:
    {instruction}
    """
    if instruction and instruction.strip():
        print(f"{instruction}")
    else:
        print("ниче нет")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": dataset_info},
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "execute_python",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                },
            },
        }
    ]

    client = get_client()

    response1 = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0.3,
    )
    msg1 = response1.choices[0].message
    messages.append(msg1)

    all_images = []

    if msg1.tool_calls:
        tool_call = msg1.tool_calls[0]
        args = json.loads(tool_call.function.arguments)
        code = args.get("code", "")
        result = execute_python(code, df)
        debug_text = (
            result.get("text_output", "") + "\n" + (result.get("error") or "")
        )
        all_images = result.get("images", [])

        if not all_images:
            fallback_images = fallback_plots(df)
            all_images.extend(fallback_images)

        safe_result = {
            "text_output": result.get("text_output", "")[:3000],
            "error": result.get("error"),
            "images_count": len(result.get("images", [])),
        }

        tool_msg = {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(safe_result, ensure_ascii=False),
        }
        messages.append(tool_msg)
    else:
        fallback_images = fallback_plots(df)
        all_images = fallback_images

    messages.append(
        {
            "role": "system",
            "content": "Теперь напиши итоговый отчёт используя полученный вывод (text_output) и информацию об изображениях",
        }
    )
    response2 = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.5,
    )
    report = response2.choices[0].message.content

    return report, all_images, debug_text
