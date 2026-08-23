import json
import threading
from types import SimpleNamespace

import pytest

import tools.persona_evaluator as ev


# 用一段固定 JSON 模拟 LLM 返回，覆盖「标识抽取」与「评分」两类解析所需的全部字段，
# 从而在不调用真实 API 的情况下验证评估逻辑（中位数聚合 / 置信度 / 反奉承锚点字段等）。
CANNED = json.dumps({
    "catches": ["喵"],
    "required_facts": ["名字小美"],
    "style_keywords": ["叠词"],
    "consistency_score": 7,
    "style_score": 6,
    "identity_score": 8,
    "overall_score": 7,
    "evidence_hit": ["引用了叠词"],
    "evidence_miss": ["未提及恐龙"],
    "persona_lift": 2,
    "reason": "回答体现了人格特色",
    "suggestion": "可加强恐龙元素",
    "meets_expectation": True,
})


class _FakeLLM:
    """模拟 ChatDeepSeek：invoke 返回构造时给定的文本（默认 CANNED）。"""

    def __init__(self, content=CANNED):
        self._content = content

    def invoke(self, prompt):
        r = SimpleNamespace()
        r.content = self._content
        return r

    def stream(self, prompt):
        yield self._content


@pytest.fixture(autouse=True)
def _clear_evaluator_caches():
    """评估器带进程内标识/基线缓存；测试间必须清空，避免相同 persona_prompt 串味。"""
    ev.clear_caches()
    yield
    ev.clear_caches()


def test_evaluate_persona_offline(monkeypatch):
    # 把所有 get_llm 调用替换成假客户端，彻底离线
    monkeypatch.setattr(ev, "get_llm", lambda *a, **k: _FakeLLM())

    out = json.loads(ev.evaluate_persona.invoke({
        "persona_prompt": "你是小美，一个 5 岁女孩",
        "user_input": "你叫什么名字？",
        "toy_response": "我叫小美，最喜欢恐龙啦~",
        "expected_behavior": "介绍自己叫小美",
    }))

    # 综合分（3 次完全一致 → 中位数 7）
    assert out["overall_score"] == 7
    # 3 次评分完全一致 → 波动 0 → 置信度高
    assert out["confidence"] == "high"
    # 反奉承锚点字段被保留
    assert out["persona_lift"] == 2
    assert out["evidence_hit"]
    # 期望行为达成判定（3 次均为 true → 通过）
    assert out["meets_expectation"] is True
    # 固定维度齐全
    for k in ("consistency_score", "style_score", "identity_score"):
        assert k in out
    # 元信息
    assert out["valid_judges"] == 3
    assert out["markers_extracted"] is True


def test_no_forced_differentiation(monkeypatch):
    """三维度真实相同时不再被强制扣分（原 _force_differentiate 会把最后一个维度 -1，
    造成「角色身份」维度被系统性压低）。"""
    flat = json.dumps({
        "catches": [], "required_facts": [], "style_keywords": [],
        "consistency_score": 7, "style_score": 7, "identity_score": 7,
        "overall_score": 7,
        "evidence_hit": [], "evidence_miss": [], "persona_lift": 1,
        "reason": "r", "suggestion": "s", "meets_expectation": True,
    })
    monkeypatch.setattr(ev, "get_llm", lambda *a, **k: _FakeLLM(flat))
    out = json.loads(ev.evaluate_persona.invoke({
        "persona_prompt": "P-flat", "user_input": "q", "toy_response": "a",
    }))
    assert out["consistency_score"] == out["style_score"] == out["identity_score"] == 7
    assert out["overall_score"] == 7


def test_hard_rule_normalization(monkeypatch):
    """硬规则匹配规范化：引号形态、全角数字差异不再产生假阴性。"""
    resp_json = json.dumps({
        "catches": ["愿\u2018圣光\u2019护佑我们"],   # 弯单引号
        "required_facts": ["阿尔德里克", "27岁"],
        "style_keywords": [],
        "consistency_score": 7, "style_score": 6, "identity_score": 8, "overall_score": 7,
        "evidence_hit": [], "evidence_miss": [], "persona_lift": 1,
        "reason": "r", "suggestion": "s", "meets_expectation": True,
    })
    monkeypatch.setattr(ev, "get_llm", lambda *a, **k: _FakeLLM(resp_json))
    out = json.loads(ev.evaluate_persona.invoke({
        "persona_prompt": "圣武士阿尔德里克，27岁，口头禅：愿圣光护佑我们",
        "user_input": "q",
        # 回答用的是弯双引号 + 全角数字，与标识形态不同
        "toy_response": "阿尔德里克说：愿\u201c圣光\u201d护佑我们！今年２７岁了。",
    }))
    hr = out["hard_rules"]
    assert hr["catches_hit"] and not hr["catches_miss"]   # 引号形态不同也命中
    assert "阿尔德里克" in hr["facts_hit"]
    assert "27岁" in hr["facts_hit"]                       # 全角２７岁 → 27岁 命中
    assert not hr["facts_miss"]


def test_majority_vote_with_single_valid_judge(monkeypatch):
    """有效评委不足 3 个：1 票也算过半（此前写死 >=2 会让只剩 1 个评委时永远判 False），
    且置信度降级为 low。"""
    lock = threading.Lock()
    state = {"n": 0}

    def flaky_judge(*a, **k):
        with lock:
            state["n"] += 1
            first = state["n"] == 1
        return json.loads(CANNED) if first else None  # 只有 1 个评委返回有效 JSON

    monkeypatch.setattr(ev, "_judge_once", flaky_judge)
    monkeypatch.setattr(ev, "get_llm", lambda *a, **k: _FakeLLM())
    out = json.loads(ev.evaluate_persona.invoke({
        "persona_prompt": "P-single", "user_input": "q", "toy_response": "a",
        "expected_behavior": "保持人设",
    }))
    assert out["valid_judges"] == 1
    assert out["confidence"] == "low"           # 评委不足 3 个 → 降级
    assert out["meets_expectation"] is True     # 1 票 * 2 > 1 → 过半


def test_markers_extraction_failure_flagged(monkeypatch):
    """标识抽取失败不再静默：输出 markers_extracted=False，全评委失败走兜底分支。"""
    monkeypatch.setattr(ev, "get_llm", lambda *a, **k: _FakeLLM("这不是JSON"))
    out = json.loads(ev.evaluate_persona.invoke({
        "persona_prompt": "P-badjson", "user_input": "q", "toy_response": "a",
    }))
    assert out["markers_extracted"] is False
    assert out["valid_judges"] == 0
    assert "评分失败" in out["reason"]
