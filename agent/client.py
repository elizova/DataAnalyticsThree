from openai import OpenAI

_client = None


def init_client(api_key: str):
    global _client
    _client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )


def get_client():
    if _client is None:
        raise RuntimeError("Клтнт не инициализирован")
    return _client
