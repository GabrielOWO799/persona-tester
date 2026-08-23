# tools/persona_evaluator.py
"""Tier-2 评估系统：
  1) 抽取人格标识（口头禅/必现事实/风格关键词）；失败时显式标记 markers_extracted=False
  2) 生成"普通助手"基线回答
  3) 硬规则：关键词命中率检查（NFKC 规范化匹配，引号/全半角/大小写差异不产生假阴性）
  4) 多评判（3 次，temp=0.4）→ 中位数聚合 + 范围/置信度（评委不足 3 个时置信度降级）
  5) 反奉承评分锚点 + 证据优先（各维度独立评分，不强制差异化）

性能：①②并行执行，③中 3 次评委并行（单条评估从 5 轮串行往返降到 2 轮）；
      标识抽取与基线回答带进程内缓存（temp=0 近似确定），回归套件重复运行大幅省调用。

返回 JSON 含：维度分（中位数）、overall_score（中位数）、score_range、confidence、
evidence_hit、evidence_miss、persona_lift、hard_rules（catches/facts 命中/缺失）、
baseline_answer、reason、suggestion、valid_judges（有效评委数）、
markers_extracted（标识抽取是否成功，False 表示硬规则输入缺失）。
"""
import hashlib
import json
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from langchain_core.tools import tool
import tools.llm_config as llm_config
from tools.llm_config import get_llm

FIXED_DIMS = [
    ("consistency_score", "人格一致性：回答是否体现人格描述中的性格、爱好、口头禅等"),
    ("style_score", "语言风格：回答的语气、用词是否与人格相符"),
    ("identity_score", "角色身份：是否明确体现了年龄、名字、身份等"),
]

# 评分锚点（反奉承 + 校准）
RUBRIC = """【评分锚点】（务必参照，不要凭印象打分）
- 0-2 分：完全没有体现人格，任何普通助手都能给这种回答
- 3-4 分：仅命中 1 个设定元素，且语气/用词仍偏通用
- 5-6 分：命中 2-3 个设定元素，但语气或细节仍偏通用
- 7-8 分：命中 4+ 个设定元素，语气、词汇、句式明显带人格特色
- 9-10 分：极度贴合，几乎像真人在说话，且无内部矛盾

【反奉承守则】
- 你是苛刻的 QA，不是鼓励师。如果回答是任何 AI 都能给的，分 ≤5。
- 大多数回答应在 4-7 区间；9+ 极少。
- 各维度独立评估：若你发现所有维度都想打同一分，请重新审视是否忽略了更弱的维度；基于证据确实相当的，可以给相同分。"""


# ---------- 抽取人格标识 ----------
_EXTRACT_PROMPT_TAIL = """\n\n人格设定：
"""  # 占位，下面继续拼接

EXTRACT_PROMPT_HEAD = """你是一个严谨的人格设定分析器。从下面的人格设定中提取可用于硬规则检查的"标识"。
只输出 JSON，结构严格如下（不要任何额外文字、注释、Markdown）：
{
  "catches": ["愿圣光护佑我们"],
  "required_facts": ["阿尔德里克", "27岁", "圣武士"],
  "style_keywords": ["哼哼唧唧"]
}
规则：
- 每个标识必须是可以直接在回答文本里找到的「裸字符串」：不要加任何前缀、注释或引号
  （错误示例："名字'阿尔德里克'"、"职业：圣武士"；正确示例："阿尔德里克"、"圣武士"）
- catches 是口头禅或固定句式，尽量保留设定中的原文片段
- required_facts 是具体可识别的事实（名字/年龄/职业/种族等），每个事实一个独立条目
- style_keywords 是标志性用词或表达方式（非通用词）
- 若某项为空就返回空数组 []
- 不要捏造未在设定中出现的内容
"""


# ---------- 基线回答 ----------
BASELINE_PROMPT_HEAD = """假设你是一个普通的 helpful AI 助手，没有任何人格设定。
请用 1-3 句话简洁、礼貌、通用地回答下面这个问题。
只输出回答本身，不要任何解释或前缀（如"当然可以"这类客套话后面直接给答案）。
"""


# ---------- 评判模板 ----------
_JUDGE_PROMPT_HEAD = """你是一个严厉的 AI 人格测试评审员，不是鼓励师。
"""

_JUDGE_DIM_SECTION = """

【评估维度】（每个维度给 0-10 整数分）
"""

_JUDGE_MARKERS_SECTION = """

【提取出的人格标识】（评估时务必参照硬规则检查）
"""

_JUDGE_BASELINE_SECTION = """

【对照基线】（一个普通助手会怎么回答）
"""

_JUDGE_INPUTS_SECTION = """

【人格设定】
"""

_JUDGE_QUESTION_LINE = """

【用户输入】
"""

_JUDGE_TOY_LINE = """

【玩具回答】
"""

_JUDGE_EXPECT_SECTION = """

【本用例的期望行为】
{expected}
请额外判断：玩具回答是否满足上述期望行为？在 JSON 中输出 "meets_expectation": true 或 false（仅当明显不满足时才为 false）。
"""

_JUDGE_FLOW_SECTION = """

【评估流程，务必按顺序输出 JSON 字段】
1. evidence_hit：列出 1-3 条回答中直接引用了设定元素的证据（必须引用原文片段）。
2. evidence_miss：列出 1-3 条本应体现但未体现的设定元素。
3. persona_lift：相对上面的"基线回答"，这个人格回答额外体现的"人格感"在 -3 到 +3 之间（0 = 一样；正数 = 更有特色；负数 = 反而更通用）。
4. 维度分数：每个维度 0-10 整数分，基于该维度自己的证据独立评分（维度间确实相当可以同分）；并给 overall_score（所有维度平均分取整）。
5. reason：一句简短理由（基于证据，不要空话）。
6. suggestion：一句具体可执行的改进建议。

【输出格式】只输出 JSON，不要任何额外文字、Markdown 代码块、注释：
{
  "consistency_score": 7,
  "style_score": 5,
  "identity_score": 6,
  "<自定义维度>_score": 7,
  "overall_score": 6,
  "evidence_hit": ["证据1（引用原文片段）", "证据2"],
  "evidence_miss": ["缺失1", "缺失2"],
  "persona_lift": 2,
  "reason": "...",
  "suggestion": "..."
}

注意：
- 维度分数键必须严格使用上面列出的字段名（包括自定义维度的 _score 后缀）。
- 各维度独立评分：若所有维度都想给同一分，先重查是否漏了更弱的维度；确实相当则同分。
"""


# ====================== 实现 ======================

def _extract_json(text):
    """从 LLM 输出中尽力抽取 JSON（容忍 ```json fence）。"""
    raw = text.strip()
    if raw.startswith("```"):
        # 去掉首行的 ```json 或 ```
        first_nl = raw.find("\n")
        if first_nl != -1:
            raw = raw[first_nl + 1:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()
    return json.loads(raw)


def _extract_persona_markers(persona_prompt, llm=None):
    """抽取人格标识。返回 dict 带 markers_extracted 标志：
    False 表示抽取失败（空三件套），让下游（评委提示词 / UI / 报告）能明确知道
    硬规则输入缺失，而不是误以为该人格没有可检查的标识。"""
    if llm is None:
        llm = get_llm(0)
    prompt = EXTRACT_PROMPT_HEAD + _EXTRACT_PROMPT_TAIL + persona_prompt
    resp = llm.invoke(prompt)
    try:
        m = _extract_json(resp.content)
        return {
            "markers_extracted": True,
            "catches": [str(x).strip() for x in (m.get("catches") or []) if str(x).strip()],
            "required_facts": [str(x).strip() for x in (m.get("required_facts") or []) if str(x).strip()],
            "style_keywords": [str(x).strip() for x in (m.get("style_keywords") or []) if str(x).strip()],
        }
    except Exception:
        return {"markers_extracted": False, "catches": [], "required_facts": [], "style_keywords": []}


def _generate_baseline(user_input, llm=None):
    if llm is None:
        llm = get_llm(0)
    prompt = BASELINE_PROMPT_HEAD + user_input
    try:
        return llm.invoke(prompt).content.strip()
    except Exception:
        return ""


# ====================== 结果缓存（省重复调用） ======================
# 标识抽取只取决于 (persona_prompt, 裁判模型)，基线只取决于 (user_input, 演员模型)，
# 且两者都在 temp=0 下生成、近似确定。回归套件对同一人格跑几十题时，
# 标识抽取只需真正调用 1 次；固定题库第二次运行时基线也能全部命中。
# 进程内 dict 足够（Streamlit 同进程跨 rerun 复用）；提示词文本一变 key 自然变化，
# 无需手动失效。单条目只有几十字，不做容量上限。
_MARKER_CACHE = {}
_BASELINE_CACHE = {}


def clear_caches():
    """清空标识/基线缓存（测试或需要强制重新生成时用）。"""
    _MARKER_CACHE.clear()
    _BASELINE_CACHE.clear()


def _cache_key(*parts):
    raw = "\x00".join(str(p) for p in parts)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _extract_persona_markers_cached(persona_prompt, judge_model):
    key = _cache_key("markers", persona_prompt, judge_model)
    if key not in _MARKER_CACHE:
        _MARKER_CACHE[key] = _extract_persona_markers(persona_prompt, llm=get_llm(0, model=judge_model))
    return _MARKER_CACHE[key]


def _generate_baseline_cached(user_input, actor_model):
    key = _cache_key("baseline", user_input, actor_model)
    if key not in _BASELINE_CACHE:
        _BASELINE_CACHE[key] = _generate_baseline(user_input, llm=get_llm(0, model=actor_model))
    return _BASELINE_CACHE[key]


# 硬规则匹配的规范化：标识与回答只在引号形态（'与'）、全半角、大小写、空格上
# 有差异时不再产生假阴性（此前纯 substring 会把"阿尔德里克"匹配不上"名字'阿尔德里克'"）
_QUOTE_RE = re.compile("[\u2018\u2019\u201c\u201d'\"\u300c\u300d\u300e\u300f\u300a\u300b]")
_WS_RE = re.compile(r"\s+")


def _norm(text):
    if not text:
        return ""
    s = unicodedata.normalize("NFKC", str(text))  # 全角→半角（２７岁→27岁）等
    s = _QUOTE_RE.sub("", s)
    s = _WS_RE.sub("", s)
    return s.casefold()


def _hard_rule_check(response, markers):
    resp_n = _norm(response)

    def hit(term):
        t = _norm(term)
        return bool(t) and t in resp_n

    def real(term):
        # 规范化后为空的标识（如纯引号）不参与检查，避免出现在 miss 列表里误导
        return bool(_norm(term))

    return {
        "catches_hit": [c for c in markers["catches"] if real(c) and hit(c)],
        "catches_miss": [c for c in markers["catches"] if real(c) and not hit(c)],
        "facts_hit": [f for f in markers["required_facts"] if real(f) and hit(f)],
        "facts_miss": [f for f in markers["required_facts"] if real(f) and not hit(f)],
    }


def _median(vals):
    s = sorted([int(v) for v in vals])
    n = len(s)
    if n == 0:
        return 0
    if n % 2 == 1:
        return s[n // 2]
    return round((s[n // 2 - 1] + s[n // 2]) / 2)


def _build_judge_prompt(persona_prompt, user_input, response, baseline, markers, all_dims, expected_behavior=None):
    dim_lines = "\n".join(f"- {key}（{label}）" for key, label in all_dims)
    markers_json = json.dumps(markers, ensure_ascii=False, indent=2)
    prompt = (
        _JUDGE_PROMPT_HEAD + RUBRIC
        + _JUDGE_DIM_SECTION + dim_lines
        + _JUDGE_MARKERS_SECTION + markers_json
        + _JUDGE_BASELINE_SECTION + (baseline or "（基线生成失败）")
        + _JUDGE_INPUTS_SECTION + persona_prompt
        + _JUDGE_QUESTION_LINE + user_input
        + _JUDGE_TOY_LINE + response
    )
    if expected_behavior:
        prompt += _JUDGE_EXPECT_SECTION.format(expected=expected_behavior)
    prompt += _JUDGE_FLOW_SECTION
    return prompt


def _judge_once(persona_prompt, user_input, response, baseline, markers, all_dims, expected_behavior=None, llm=None):
    """一次独立评分（temp=0.4 拿一些变化）。"""
    if llm is None:
        llm = get_llm(0.4)
    prompt = _build_judge_prompt(persona_prompt, user_input, response, baseline, markers, all_dims, expected_behavior)
    resp = llm.invoke(prompt)
    try:
        return _extract_json(resp.content)
    except Exception:
        return None


@tool
def evaluate_persona(persona_prompt: str, user_input: str, toy_response: str, custom_dims: Optional[list] = None, expected_behavior: Optional[str] = None, judge_model: Optional[str] = None) -> str:
    """评估玩具的回答是否符合人格设定。
    返回 JSON：各维度分（中位数）、overall_score（中位数）、score_range、confidence、
    evidence_hit/miss、persona_lift、hard_rules（catches/facts 命中/缺失）、baseline_answer、reason、suggestion、
    valid_judges（有效评委数，不足 3 个时置信度会降级）、markers_extracted（标识抽取是否成功，
    False 表示硬规则输入缺失，分数与证据可能有偏差）。
    若提供 expected_behavior，额外返回 meets_expectation（有效评委过半数的 pass/fail），供测试套件判定用例通过与否。
    judge_model 可选：指定评分用的模型（如 'deepseek-reasoner'），与演员模型解耦，避免「裁判=演员」同一大脑。
    为 None 时沿用全局演员模型（向后兼容）。
    """
    # 1. 准备维度
    custom_dims = [d.strip() for d in (custom_dims or []) if d and d.strip()]
    all_dims = list(FIXED_DIMS)
    for d in custom_dims:
        all_dims.append((f"{d}_score", d))
    all_dims_keys = [k for k, _ in all_dims]

    # 2. 抽取人格标识 + 生成基线 + 硬规则
    # 裁判侧分析（标识抽取）用裁判模型；基线"普通助手"用演员模型（保证 persona_lift 只反映人格差异，不反映模型差异）
    # 标识抽取与基线互相独立（前者只看 persona_prompt，后者只看 user_input），并行省一轮往返；
    # 两者各自带缓存，同一人格的回归套件只会真正抽取一次标识。
    # 注意 judge_model 标签必须在调用时读 llm_config.MODEL（模块级快照会拿到切换前的旧值）
    _judge_model = judge_model or llm_config.MODEL
    _actor_model = llm_config.MODEL
    with ThreadPoolExecutor(max_workers=3) as pool:
        fm = pool.submit(_extract_persona_markers_cached, persona_prompt, judge_model)
        fb = pool.submit(_generate_baseline_cached, user_input, _actor_model)
        markers = fm.result()
        baseline = fb.result()
        hr = _hard_rule_check(toy_response, markers)

        # 3. 多评判（3 次并行，均使用裁判模型，与演员解耦；单个评委失败不拖垮整体）
        judge_llm = get_llm(0.4, model=judge_model)
        judges = []
        jf = [
            pool.submit(_judge_once, persona_prompt, user_input, toy_response,
                        baseline, markers, all_dims, expected_behavior, llm=judge_llm)
            for _ in range(3)
        ]
        for fut in jf:
            try:
                j = fut.result()
            except Exception:
                j = None
            if j:
                judges.append(j)

    if not judges:
        # 全部失败的兜底
        out = {k: 0 for k in all_dims_keys}
        out["overall_score"] = 0
        out.update({
            "score_range": 0, "confidence": "low",
            "evidence_hit": [], "evidence_miss": [], "persona_lift": 0,
            "hard_rules": hr, "baseline_answer": baseline,
            "reason": "（评分失败：所有评判均未返回有效 JSON）",
            "suggestion": "请稍后重试，或临时减少自定义维度数量。",
            "judge_model": _judge_model,
            "valid_judges": 0,
            "markers_extracted": bool(markers.get("markers_extracted", True)),
        })
        return json.dumps(out, ensure_ascii=False, indent=2)

    # 4. 中位数聚合
    agg = {}
    ranges = {}
    for key in all_dims_keys:
        vals = []
        for j in judges:
            try:
                v = int(j.get(key, 0))
                vals.append(v)
            except Exception:
                continue
        agg[key] = _median(vals) if vals else 0
        ranges[key] = (max(vals) - min(vals)) if vals else 0
    avg_range = round(sum(ranges.values()) / len(ranges), 1) if ranges else 0.0

    # 5. 置信度：评委不足 3 个时统计基础薄弱，直接降级；否则按维度平均波动
    if len(judges) < 3:
        confidence = "low"
    elif avg_range <= 1:
        confidence = "high"
    elif avg_range <= 2:
        confidence = "medium"
    else:
        confidence = "low"

    # 6. 综合分（各维度中位数取整平均；各维度独立评分，不做事后强制差异化）
    agg["overall_score"] = round(
        sum(agg[k] for k in all_dims_keys) / max(1, len(all_dims_keys))
    )

    # 7. 证据 / 缺失 / lift（取整体分最接近 median 的那次评判）
    median_overall = agg["overall_score"]
    def _dist(j):
        try:
            return abs(int(j.get("overall_score", 0)) - median_overall)
        except Exception:
            return 999
    closest = min(judges, key=_dist)
    agg["evidence_hit"] = closest.get("evidence_hit", []) or []
    agg["evidence_miss"] = closest.get("evidence_miss", []) or []
    try:
        agg["persona_lift"] = int(round(float(closest.get("persona_lift", 0))))
    except Exception:
        agg["persona_lift"] = 0

    # 8. reason / suggestion（来自 closest）
    agg["reason"] = closest.get("reason", "（无）") or "（无）"
    agg["suggestion"] = closest.get("suggestion", "（无）") or "（无）"

    # 9. 元信息
    agg["score_range"] = avg_range
    agg["confidence"] = confidence
    agg["hard_rules"] = hr
    agg["baseline_answer"] = baseline
    agg["judge_model"] = _judge_model
    agg["valid_judges"] = len(judges)
    agg["markers_extracted"] = bool(markers.get("markers_extracted", True))

    # 10. 期望行为达成判定（有效评委过半数为真 → 通过）
    # 评委可能因 JSON 解析失败不足 3 个：1 个评委时 1 票即过半，2 个时需全票，3 个时 2 票。
    # 此前写死 >=2 会在只剩 1 个有效评委时永远判 False。
    def _truthy(v):
        return str(v).strip().lower() in ("true", "1", "yes", "是")
    if expected_behavior:
        n_true = sum(1 for j in judges if _truthy(j.get("meets_expectation")))
        agg["meets_expectation"] = n_true * 2 > len(judges)

    return json.dumps(agg, ensure_ascii=False, indent=2)