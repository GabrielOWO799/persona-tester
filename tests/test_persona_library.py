import tools.persona_library as lib


def _redirect(monkeypatch, tmp_path):
    """把人格库读写重定向到临时目录，避免污染真实 data/。"""
    user_file = tmp_path / "user_personas.json"
    ref_file = tmp_path / "reference_personas.json"
    monkeypatch.setattr(lib, "USER_FILE", str(user_file))
    monkeypatch.setattr(lib, "REFERENCE_FILE", str(ref_file))
    return user_file


def test_add_and_get_user_persona_with_versions(tmp_path, monkeypatch):
    _redirect(monkeypatch, tmp_path)
    v1 = lib.add_user_persona("测试人格", "你是一个测试助手。")
    assert v1 == 1
    p = lib.get_persona_by_name("测试人格")
    assert p["persona"] == "你是一个测试助手。"
    assert p["version"] == 1

    # 同一人格再次保存应新增版本（v2），而非覆盖
    v2 = lib.add_user_persona("测试人格", "你是升级版测试助手。")
    assert v2 == 2
    p2 = lib.get_persona_by_name("测试人格", version=2)
    assert p2["persona"] == "你是升级版测试助手。"
    # 默认返回最新版本
    latest = lib.get_persona_by_name("测试人格")
    assert latest["persona"] == "你是升级版测试助手。"


def test_test_cases_roundtrip(tmp_path, monkeypatch):
    _redirect(monkeypatch, tmp_path)
    lib.add_user_persona("用例人格", "提示词")
    cases = [{"input": "你好", "expected_behavior": "打招呼"}]
    assert lib.set_test_cases("用例人格", cases) is True
    assert lib.get_test_cases("用例人格") == cases


def test_unknown_persona_returns_none(tmp_path, monkeypatch):
    _redirect(monkeypatch, tmp_path)
    assert lib.get_persona_by_name("不存在的人格") is None
