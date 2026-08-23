# tools/persona_history.py
"""聊天历史的本地持久化：让刷新页面后能恢复上次的人格与对话。"""
import json
import os
import hashlib

import config

CHAT_HISTORY_DIR = config.CHAT_HISTORY_DIR
LAST_ACTIVE_FILE = config.LAST_ACTIVE_PATH


def _safe_key(name: str) -> str:
    """人格名可能含中文/特殊字符，用 md5 做安全且唯一的文件名。"""
    return hashlib.md5(name.encode("utf-8")).hexdigest()


def load_chat_history(name: str):
    """读取某人格的聊天记录，不存在则返回空列表。"""
    path = os.path.join(CHAT_HISTORY_DIR, f"{_safe_key(name)}.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_chat_history(name: str, messages):
    """保存某人格的聊天记录到本地。"""
    os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)
    path = os.path.join(CHAT_HISTORY_DIR, f"{_safe_key(name)}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)


def save_last_active(persona: dict):
    """记录"上次活跃的人格"（含提示词快照），供刷新后自动恢复。"""
    os.makedirs(config.DATA_DIR, exist_ok=True)
    with open(LAST_ACTIVE_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "name": persona.get("name"),
            "persona": persona.get("persona"),
            "version": persona.get("version"),
        }, f, ensure_ascii=False, indent=2)


def load_last_active():
    """读取上次活跃人格；不存在或异常返回 None。"""
    if not os.path.exists(LAST_ACTIVE_FILE):
        return None
    try:
        with open(LAST_ACTIVE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None
