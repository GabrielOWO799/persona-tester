import json
import os
from datetime import datetime

import config

REFERENCE_FILE = config.REFERENCE_PERSONAS_PATH
USER_FILE = config.USER_PERSONAS_PATH

def load_reference_personas():
    if not os.path.exists(REFERENCE_FILE):
        return []
    with open(REFERENCE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def load_user_personas():
    if not os.path.exists(USER_FILE):
        return []
    with open(USER_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_user_personas(personas):
    with open(USER_FILE, "w", encoding="utf-8") as f:
        json.dump(personas, f, ensure_ascii=False, indent=2)

def add_user_persona(name, persona, note=""):
    personas = load_user_personas()
    existing = None
    for p in personas:
        if p["name"] == name:
            existing = p
            break

    if existing:
        versions = existing.get("versions", [])
        new_version_num = len(versions) + 1
        versions.append({
            "version": new_version_num,
            "persona": persona,
            "created_at": datetime.now().isoformat(),
            "note": note
        })
        existing["versions"] = versions
        existing["persona"] = persona  # 更新顶层字段为最新版本
    else:
        new_version_num = 1
        new_entry = {
            "name": name,
            "persona": persona,
            "versions": [{
                "version": 1,
                "persona": persona,
                "created_at": datetime.now().isoformat(),
                "note": note
            }],
            "test_cases": []
        }
        personas.append(new_entry)
    save_user_personas(personas)
    return new_version_num

def get_all_personas():
    ref = load_reference_personas()
    ref_list = [{"name": p["name"], "persona": p["persona"], "is_reference": True} for p in ref]

    user = load_user_personas()
    user_list = []
    for p in user:
        if "versions" in p and p["versions"]:
            latest = p["versions"][-1]
            user_list.append({
                "name": p["name"],
                "persona": latest["persona"],
                "is_reference": False,
                "versions": p["versions"]
            })
        else:
            # 兼容没有 versions 的旧数据
            user_list.append({
                "name": p["name"],
                "persona": p["persona"],
                "is_reference": False,
                "versions": []
            })
    return ref_list + user_list

def get_persona_by_name(name, version=None):
    # 先查参考库
    ref = load_reference_personas()
    for p in ref:
        if p["name"] == name:
            return {"name": name, "persona": p["persona"], "is_reference": True}
    # 再查用户库（修正：使用 load_user_personas）
    user = load_user_personas()
    for p in user:
        if p["name"] == name:
            versions = p.get("versions", [])
            if not versions:
                # 没有版本信息，直接返回顶层 persona
                return {
                    "name": name,
                    "persona": p["persona"],
                    "is_reference": False,
                    "version": 1,
                    "versions": []
                }
            if version is None:
                latest = versions[-1]
                return {
                    "name": name,
                    "persona": latest["persona"],
                    "is_reference": False,
                    "version": latest["version"],
                    "versions": versions
                }
            else:
                for v in versions:
                    if v["version"] == version:
                        return {
                            "name": name,
                            "persona": v["persona"],
                            "is_reference": False,
                            "version": v["version"],
                            "versions": versions
                        }
                return None
    return None

def update_persona_version(name, new_persona, note=""):
    return add_user_persona(name, new_persona, note)

def get_test_cases(name):
    """读取某人格的测试用例（参考库 / 用户库均支持）。"""
    ref = load_reference_personas()
    for p in ref:
        if p["name"] == name:
            return p.get("test_cases", [])
    user = load_user_personas()
    for p in user:
        if p["name"] == name:
            return p.get("test_cases", [])
    return []

def set_test_cases(name, cases):
    """覆写某人格的测试用例，返回是否成功写入。"""
    ref = load_reference_personas()
    for p in ref:
        if p["name"] == name:
            p["test_cases"] = cases
            with open(REFERENCE_FILE, "w", encoding="utf-8") as f:
                json.dump(ref, f, ensure_ascii=False, indent=2)
            return True
    user = load_user_personas()
    for p in user:
        if p["name"] == name:
            p["test_cases"] = cases
            save_user_personas(user)
            return True
    return False