import httpx

from .config import settings


class LLMError(RuntimeError):
    pass


async def generate(system_prompt: str, user_prompt: str) -> str:
    payload = {
        "model": settings.ollama_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.1, "num_ctx": 4096},
    }
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            r = await client.post(f"{settings.ollama_url}/api/chat", json=payload)
            r.raise_for_status()
            return r.json()["message"]["content"].strip()
    except (httpx.HTTPError, KeyError, ValueError) as e:
        raise LLMError(f"LLM unavailable: {e}") from e