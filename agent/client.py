from langchain_openai import ChatOpenAI

_client = None


def init_client(api_key: str):
    global _client
    _client = ChatOpenAI(
        model="llama-3.1-8b-instant",  # model="llama-3.3-70b-versatile",
        openai_api_key=api_key,
        openai_api_base="https://api.groq.com/openai/v1",
        temperature=0.3,
    )


def get_client():
    if _client is None:
        raise RuntimeError("Клтнт не инициализирован")
    return _client
