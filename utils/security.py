def sanitize_instruction(text: str) -> str:
    forbidden = ["import os", "sys.", "open(", "exec(", "eval("]
    text_lower = text.lower()

    for f in forbidden:
        if f in text_lower:
            return ""

    return text
