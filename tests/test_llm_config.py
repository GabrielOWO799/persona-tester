import os

import tools.llm_config as cfg


def test_cache_reuses_same_instance():
    os.environ.setdefault("DEEPSEEK_API_KEY", "dummy")
    a = cfg.get_llm(retry=False)
    b = cfg.get_llm(retry=False)
    assert a is b


def test_cache_distinguishes_model_and_temperature():
    os.environ.setdefault("DEEPSEEK_API_KEY", "dummy")
    d = cfg.get_llm(0, retry=False)
    e = cfg.get_llm(0, retry=False)
    f = cfg.get_llm(0.4, retry=False)
    g = cfg.get_llm(0.4, model="deepseek-reasoner", retry=False)
    assert d is e
    assert d is not f
    assert f is not g


def test_set_model_and_temperature_apply():
    cfg.set_model("deepseek-chat")
    cfg.set_temperature(0.7)
    assert cfg.MODEL == "deepseek-chat"
    assert cfg.TEMPERATURE == 0.7
