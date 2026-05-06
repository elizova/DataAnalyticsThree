from langchain_openai import ChatOpenAI

_client = None


def init_client(api_key: str):
    global _client
    _client = ChatOpenAI(
        model="meta-llama/llama-3.3-70b-instruct",
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.3,
    )


def get_client():
    if _client is None:
        raise RuntimeError("Клтнт не инициализирован")
    return _client
