"""Мок AD: чтение пользователей и групп из JSON. В проде нужен OIDC/LDAP."""
import json
from functools import lru_cache
from pathlib import Path

from fastapi import Header, HTTPException

from .config import settings


@lru_cache
def _load_users() -> dict:
    return json.loads(Path(settings.users_file).read_text(encoding="utf-8"))


def get_user_by_id(user_id: str) -> dict | None:
    for u in _load_users()["users"]:
        if u["id"] == user_id:
            return u
    return None


async def current_user(x_user_id: str = Header(..., alias="X-User-Id")) -> dict:
    """
    Для прототипа аутентификация это заголовок X-User-Id (его может подделать кто угодно).
    В продакшене нужна валидация JWT/OIDC и получение групп из AD на стороне сервера.
    """
    user = get_user_by_id(x_user_id)
    if not user:
        raise HTTPException(status_code=401, detail=f"Unknown user: {x_user_id}")
    return user