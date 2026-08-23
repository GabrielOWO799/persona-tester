"""统一执行引擎 run_case / run_cases 的离线测试（注入假 chat/eval，零 API 调用）。"""
import json
from types import SimpleNamespace

import pytest

import test_suite as ts


def _fake_chat(args):
    q = args.get("user_input", "")
    return SimpleNamespace(content=f"回答[{q}]")


def _fake_eval(args):
    out = {
        "consistency_score": 7,
        "style_score": 6,
        "identity_score": 8,
        "overall_score": 7,
        "evidence_miss": ["缺了恐龙"],
        "suggestion": "加强恐龙元素",
        "meets_expectation": True,
    }
    return SimpleNamespace(content=json.dumps(out, ensure_ascii=False))


def test_run_cases_happy_path_ordered():
    cases = [{"input": f"问题{i}", "expected_behavior": "保持人设"} for i in range(6)]
    rows = ts.run_cases("人格P", cases, chat_fn=_fake_chat, eval_fn=_fake_eval, max_workers=3)
    # 并发下完成序是乱序的，但结果必须保持输入顺序（报告行号/基线 id 依赖这一点）
    assert [r["input"] for r in rows] == [c["input"] for c in cases]
    assert [r["id"] for r in rows] == list(range(6))
    assert all(r["error"] is None for r in rows)
    assert ts.row_score(rows[0]) == 7
    assert ts.row_passed(rows[0]) is True
    assert rows[0]["scores"]["identity_score"] == 8
    assert rows[0]["eval"]["suggestion"] == "加强恐龙元素"


def test_run_cases_builtin_question_format():
    # 内置题库格式 {category, question, expected}；空输入是合法探针，不跳过
    cases = [
        {"category": "身份挑战", "question": "你是谁？", "expected": "坚持人设"},
        {"category": "极端输入", "question": "", "expected": "得体回应"},
    ]
    rows = ts.run_cases("人格P", cases, chat_fn=_fake_chat, eval_fn=_fake_eval, max_workers=1)
    assert rows[0]["category"] == "身份挑战"
    assert rows[0]["input"] == "你是谁？"
    assert rows[0]["expected"] == "坚持人设"
    assert rows[1]["input"] == ""


def test_run_cases_no_expected_marks_passed_none():
    rows = ts.run_cases("P", [{"input": "你好"}], chat_fn=_fake_chat, eval_fn=_fake_eval)
    # 未设期望行为的用例无法判定 pass/fail，不参与合格率统计
    assert ts.row_passed(rows[0]) is None


def test_run_cases_records_errors_per_case():
    def chat(args):
        if args.get("user_input") == "chat失败":
            raise RuntimeError("网络炸了")
        return _fake_chat(args)

    def eval_fn(args):
        if args.get("user_input") == "eval失败":
            raise ValueError("评估挂了")
        return _fake_eval(args)

    cases = [{"input": "正常题", "expected_behavior": "x"},
             {"input": "chat失败", "expected_behavior": "x"},
             {"input": "eval失败", "expected_behavior": "x"}]
    rows = ts.run_cases("P", cases, chat_fn=chat, eval_fn=eval_fn, max_workers=1)

    assert rows[0]["error"] is None and ts.row_score(rows[0]) == 7
    # 对话失败：记录原因、无回答无分数、不计合格率
    assert "网络炸了" in rows[1]["error"]
    assert rows[1]["response"] == ""
    assert ts.row_score(rows[1]) is None and ts.row_passed(rows[1]) is None
    # 评估失败：回答保留，error 记录原因
    assert "评估挂了" in rows[2]["error"]
    assert rows[2]["response"].startswith("回答[eval失败]")
    assert ts.row_score(rows[2]) is None


def test_run_cases_bad_eval_json_records_error():
    def bad_eval(args):
        return SimpleNamespace(content="不是JSON")

    rows = ts.run_cases("P", [{"input": "q"}], chat_fn=_fake_chat, eval_fn=bad_eval)
    assert "解析评估 JSON 失败" in rows[0]["error"]
    assert ts.row_score(rows[0]) is None


def test_run_regression_suite_mock_writes_baseline_and_diffs(tmp_path):
    bl, rp = tmp_path / "bl.json", tmp_path / "rp.md"
    qs = ts.CORE_QUESTIONS[:4]
    rows, report = ts.run_regression_suite("P", qs, baseline_path=str(bl), report_path=str(rp),
                                           mock=True, max_workers=2)
    assert len(rows) == 4
    assert all(r["error"] is None for r in rows)
    assert "回归评测报告" in report
    assert bl.exists() and rp.exists()
    # 第二次运行：读到基线 → 报告出现波动对比（mock 分数确定性 → 持平）
    _, report2 = ts.run_regression_suite("P", qs, baseline_path=str(bl), report_path=str(rp), mock=True)
    assert "→ 持平" in report2
    assert "无（首次运行" not in report2


def test_run_regression_suite_resume_preserves_prior_entries(tmp_path):
    # 回归测试：基线 key 是字符串、运行时 id 是整数，两者混用曾导致 resume 把旧条目静默丢弃
    bl, rp = tmp_path / "bl.json", tmp_path / "rp.md"
    qs = ts.CORE_QUESTIONS[:4]
    ts.run_regression_suite("P", qs[:2], baseline_path=str(bl), report_path=str(rp), mock=True)
    assert len(json.load(open(bl, encoding="utf-8"))["results"]) == 2
    # resume 跑全 4 条：前 2 条已在基线（跳过不重跑），后 2 条补跑，最终基线应为 4 条
    rows, _ = ts.run_regression_suite("P", qs, baseline_path=str(bl), report_path=str(rp),
                                      mock=True, resume=True)
    assert len(rows) == 4
    assert len(json.load(open(bl, encoding="utf-8"))["results"]) == 4


def test_run_regression_suite_all_failed_raises(tmp_path, monkeypatch):
    def boom(args):
        raise RuntimeError("API 挂了")

    monkeypatch.setattr(ts, "_mock_fns", lambda: (boom, boom))
    with pytest.raises(RuntimeError, match="API 挂了"):
        ts.run_regression_suite("P", ts.CORE_QUESTIONS[:2],
                                baseline_path=str(tmp_path / "b.json"),
                                report_path=str(tmp_path / "r.md"), mock=True)
