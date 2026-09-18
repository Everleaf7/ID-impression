from __future__ import annotations

import re


KNOWN_WORDS = ("自由人", "玫瑰", "果汁", "萝莉", "春花", "春", "花", "梦", "龙")


def tokenize_id(user_id: str) -> list[str]:
    value = user_id.strip()
    if not value:
        raise ValueError("ID / 昵称不能为空")

    tokens: list[str] = []
    remaining = value
    for word in KNOWN_WORDS:
        if word in remaining:
            prefix, remaining = remaining.split(word, 1)
            tokens.extend(_basic_split(prefix))
            tokens.append(word)
    tokens.extend(_basic_split(remaining))
    return [token for token in tokens if token]


def _basic_split(value: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]|[^\s]", value)
