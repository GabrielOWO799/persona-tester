# app.py
import streamlit as st
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from tools.persona_generator import generate_persona
from tools.persona_chat import persona_chat, stream_persona_response
from tools.persona_evaluator import evaluate_persona
from tools.persona_library import (
    get_persona_by_name, get_all_personas,
    get_test_cases, set_test_cases,
)
from tools.persona_optimizer import optimize_persona
from tools.persona_history import (
    load_chat_history, save_chat_history,
    save_last_active, load_last_active,
)
import difflib
import tools.llm_config as llm_config
from tools.report_export import to_markdown, to_pdf, iter_dimensions
import config
from test_suite import (
    run_cases, run_regression_suite, row_score, row_passed,
    CORE_QUESTIONS, FULL_QUESTIONS, DEFAULT_WORKERS,
)

# 确保运行时目录存在（含「曾被误建为文件」的修复，见 config.ensure_dirs）
config.ensure_dirs()
REPORTS_DIR = config.REPORTS_DIR


def safe_tool(tool, kwargs, label="操作"):
    """_try_invoke 的便捷封装：只返回结果，失败返回 None。
    需要同时拿到错误信息（如测试套件要记录失败原因）时，请直接用 _try_invoke。"""
    result, _err = _try_invoke(tool, kwargs, label)
    return result


def _last_assistant(messages):
    """取对话里最后一条玩具（assistant）回复内容。"""
    for m in reversed(messages):
        if m.get("role") == "assistant":
            return m.get("content", "")
    return ""


def _render_version_diff(vd):
    """主区域渲染：并排显示某人格两个版本的提示词 + 行级差异。"""
    name = vd["name"]
    pa = get_persona_by_name(name, version=vd["va"])
    pb = get_persona_by_name(name, version=vd["vb"])
    if not pa or not pb:
        st.error("找不到对应版本")
        return
    st.subheader(f"🔍 版本对比：{name}")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**版本 A（v{vd['va']}）**")
        st.text_area("提示词 A", pa["persona"], height=320, disabled=True)
    with c2:
        st.markdown(f"**版本 B（v{vd['vb']}）**")
        st.text_area("提示词 B", pb["persona"], height=320, disabled=True)
    diff = "\n".join(difflib.unified_diff(
        pa["persona"].splitlines(), pb["persona"].splitlines(),
        fromfile=f"A v{vd['va']}", tofile=f"B v{vd['vb']}", lineterm=""))
    st.markdown("**差异（unified diff，红 `-`=A 独有，绿 `+`=B 独有）**")
    st.code(diff, language="diff")


def _try_invoke(tool, kwargs, label):
    """统一包裹 LLM 工具调用的核心实现：出错时给出友好提示而非白屏，并返回 (结果, 错误信息)。
    返回元组便于调用方区分成功 / 失败（测试套件需记录失败原因）。safe_tool 是其便捷封装。"""
    try:
        return tool.invoke(kwargs), None
    except Exception as e:
        msg = f"{label}失败：{e}"
        st.error(f"⚠️ {msg}\n请检查网络或 API Key 后重试。")
        return None, msg


def run_regression(name, persona_a, persona_b, test_cases, custom_dims=None, judge_model=None,
                   progress_callback=None):
    """同一套测试用例，分别跑 vA / vB，逐条对比得分 delta。
    执行引擎复用 test_suite.run_cases（并发执行、逐条记录失败，与测试套件/一键回归口径一致）。"""
    phase = {"label": "vA"}

    def _p(i, total, c):
        if progress_callback:
            progress_callback(i, total, c, phase["label"])

    res_a = run_cases(persona_a, test_cases, custom_dims, judge_model, progress_callback=_p)
    phase["label"] = "vB"
    res_b = run_cases(persona_b, test_cases, custom_dims, judge_model, progress_callback=_p)

    rows = []
    n = max(len(res_a), len(res_b))
    for i in range(n):
        a = res_a[i] if i < len(res_a) else None
        b = res_b[i] if i < len(res_b) else None
        sa = row_score(a) if a else None
        sb = row_score(b) if b else None
        delta = (sb - sa) if (sa is not None and sb is not None) else None
        ref = a or b
        rows.append({
            "input": ref["input"],
            "expected": ref.get("expected", ""),
            "score_a": sa, "score_b": sb, "delta": delta,
            "pass_a": (row_passed(a) if a else None),
            "pass_b": (row_passed(b) if b else None),
        })

    def _avg(lst):
        vals = [row_score(r) for r in lst if row_score(r) is not None]
        return round(sum(vals) / len(vals), 1) if vals else 0

    def _rate(lst):
        _valid = [r for r in lst if row_passed(r) is not None]
        if not _valid:
            return 0
        return round(sum(1 for r in _valid if row_passed(r)) / len(_valid) * 100)

    def _failed(lst):
        return sum(1 for r in lst if r.get("error"))

    return {
        "name": name,
        "rows": rows,
        "avg_a": _avg(res_a), "avg_b": _avg(res_b),
        "pass_a": _rate(res_a), "pass_b": _rate(res_b),
        "failed_a": _failed(res_a), "failed_b": _failed(res_b),
    }


# 页面配置
st.set_page_config(page_title="人格测试助手", page_icon="🧸", layout="wide")
st.title("🧸 AI 人格测试助手")

# 初始化 session_state
if "current_persona" not in st.session_state:
    st.session_state.current_persona = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "comparison_mode" not in st.session_state:
    st.session_state.comparison_mode = False
if "persona_a" not in st.session_state:
    st.session_state.persona_a = None
if "persona_b" not in st.session_state:
    st.session_state.persona_b = None
if "await_feedback" not in st.session_state:
    st.session_state.await_feedback = False
if "last_evaluation" not in st.session_state:
    st.session_state.last_evaluation = ""
if "last_eval_data" not in st.session_state:
    st.session_state.last_eval_data = None
if "last_user_input" not in st.session_state:
    st.session_state.last_user_input = ""
if "version_diff" not in st.session_state:
    st.session_state.version_diff = None
if "test_suite" not in st.session_state:
    st.session_state.test_suite = None
if "test_suite_name" not in st.session_state:
    st.session_state.test_suite_name = ""
if "version_regression" not in st.session_state:
    st.session_state.version_regression = None
if "last_compare" not in st.session_state:
    st.session_state.last_compare = None
if "judge_model" not in st.session_state:
    st.session_state.judge_model = "deepseek-chat"
if "oneclick_regression" not in st.session_state:
    st.session_state.oneclick_regression = None  # 一键回归结果：{"md","name","path"} 或 None

# 刷新后恢复：若上次有活跃人格，自动载入它及其聊天历史
if st.session_state.current_persona is None:
    _last = load_last_active()
    if _last and _last.get("name") and _last.get("persona"):
        st.session_state.current_persona = {
            "name": _last["name"],
            "persona": _last["persona"],
            "version": _last.get("version"),
        }
        st.session_state.messages = load_chat_history(_last["name"])

# 加载人格库（含参考人格与用户人格，统一一个列表）
all_personas = get_all_personas()
persona_names = [p["name"] for p in all_personas]


# 侧边栏：选择参考人格
with st.sidebar:
    st.header("📚 参考人格库")
    selected_name = st.selectbox("选择已测试人格", persona_names,key="persona_select")
    temp_persona = get_persona_by_name(selected_name)
    if temp_persona and not temp_persona.get("is_reference") and len(temp_persona.get("versions", [])) > 1:
        versions=temp_persona["versions"]
        version_options = [f"v{v['version']} ({v['created_at'][:10]})" for v in versions]
        selected_version_str=st.selectbox("选择版本",version_options,key="version_select")
        #解析版本号
        selected_version=None
        for v in versions:
            if f"v{v['version']} ({v['created_at'][:10]})" == selected_version_str:
                selected_version = v['version']
                break
        if selected_version is None:
            selected_version = versions[-1]['version']
    else:
        selected_version = None

    if st.button("加载此人格"):
        if temp_persona:
            if selected_version is not None:
                persona_to_load = get_persona_by_name(selected_name, version=selected_version)
            else:
                persona_to_load = temp_persona
            st.session_state.current_persona = persona_to_load
            st.session_state.messages = load_chat_history(persona_to_load["name"])
            save_last_active(persona_to_load)
            version_info = f" (版本 v{persona_to_load.get('version', '最新')})" if not persona_to_load.get("is_reference") else ""
            st.success(f"✅ 已加载人格：{persona_to_load['name']}{version_info}")
            st.rerun()
        else:
            st.error("未找到该人格")
            
    st.divider()
    st.header("⚙️ 模型与温度")
    model_options = ["deepseek-chat", "deepseek-reasoner", "其他（自定义）"]
    model_sel = st.selectbox("模型", model_options, index=0, key="model_sel")
    if model_sel == "其他（自定义）":
        custom_model = st.text_input("自定义模型名", key="custom_model")
        model = custom_model.strip() or "deepseek-chat"
    else:
        model = model_sel
    llm_config.set_model(model)
    temp = st.slider("温度（回答/生成）", 0.0, 1.0, llm_config.TEMPERATURE, 0.05, key="temp")
    llm_config.set_temperature(temp)

    st.divider()
    st.header("⚖️ 裁判模型（评分用）")
    st.caption("默认与上方一致；改成不同模型可避免「裁判=演员」同一大脑，评分更中立（如演员用 reasoner、裁判用 chat）。")
    judge_options = ["deepseek-chat", "deepseek-reasoner", "其他（自定义）"]
    judge_sel = st.selectbox("裁判模型", judge_options, index=0, key="judge_sel")
    if judge_sel == "其他（自定义）":
        custom_judge = st.text_input("自定义裁判模型名", key="custom_judge")
        judge_model = custom_judge.strip() or "deepseek-chat"
    else:
        judge_model = judge_sel
    st.session_state.judge_model = judge_model

    st.divider()
    st.header("🎯 评估维度")
    cd_text = st.text_input("自定义维度（逗号分隔，如：幽默感,毒舌）", key="cd",
                            value=",".join(st.session_state.get("custom_dims", [])))
    st.session_state.custom_dims = [x.strip() for x in cd_text.split(",") if x.strip()]

    # ============ 工具箱（折叠）：版本对比 / 版本回归 / 测试套件 / 一键回归测试 ============
    with st.expander("🧰 工具箱（点击展开）", expanded=False):
        st.caption("把「版本对比 / 版本回归 / 测试套件 / 一键回归测试」收进工具箱，需要用时再展开。")
        st.divider()
        # 版本对比入口（仅当用户人格有 >=2 个版本时显示）
        if temp_persona and not temp_persona.get("is_reference") and len(temp_persona.get("versions", [])) >= 2:
            st.divider()
            st.header("🔍 版本对比")
            _vers = temp_persona["versions"]
            _vopts = [f"v{v['version']}（{v['created_at'][:10]}）" for v in _vers]
            _va = st.selectbox("版本 A", _vopts, key="va")
            _vb = st.selectbox("版本 B", _vopts, index=len(_vopts) - 1, key="vb")
            if st.button("查看差异", key="cmp_view"):
                def _ver_num(s): return int(s.split("v")[1].split("（")[0])
                st.session_state.version_diff = {"name": temp_persona["name"], "va": _ver_num(_va), "vb": _ver_num(_vb)}
                st.rerun()
    
        # 版本回归入口（同套用例，vA vs vB 得分 delta；需 >=2 版本且存在测试用例）
        if temp_persona and not temp_persona.get("is_reference") and len(temp_persona.get("versions", [])) >= 2:
            _reg_cases = get_test_cases(temp_persona["name"])
            if _reg_cases:
                st.divider()
                st.header("🔁 版本回归")
                _r_vers = temp_persona["versions"]
                _r_vopts = [f"v{v['version']}（{v['created_at'][:10]}）" for v in _r_vers]
                _rva = st.selectbox("回归版本 A", _r_vopts, key="rva")
                _rvb = st.selectbox("回归版本 B", _r_vopts, index=len(_r_vopts) - 1, key="rvb")
    
                def _rver_num(s): return int(s.split("v")[1].split("（")[0])
                if st.button("运行回归对比", key="reg_run"):
                    _pa = get_persona_by_name(temp_persona["name"], version=_rver_num(_rva))
                    _pb = get_persona_by_name(temp_persona["name"], version=_rver_num(_rvb))
                    if not _pa or not _pb:
                        st.error("找不到对应版本")
                    else:
                        _prog = st.progress(0.0, text="准备中…")

                        def _vp(i, total, c, label):
                            # vA/vB 各占进度条的一半
                            _frac = (0.0 if label == "vA" else 0.5) + (i / total) * 0.5
                            _q = c.get("input") or c.get("question") or ""
                            _prog.progress(min(_frac, 1.0), text=f"{label}：[{i}/{total}] {_q[:20] or '（空）'}")

                        with st.spinner("回归计算中…"):
                            _reg = run_regression(
                                temp_persona["name"], _pa["persona"], _pb["persona"],
                                _reg_cases, st.session_state.get("custom_dims", []),
                                st.session_state.judge_model, progress_callback=_vp)
                        _prog.empty()
                        st.session_state.version_regression = _reg
                        st.rerun()
    
        # 测试套件入口（为所选人格维护测试用例，一键跑出合格率）
        if temp_persona:
            _tc_name = temp_persona["name"]
            _tc = get_test_cases(_tc_name)
            # 切换人格时重新初始化可编辑列表；同一人格内保留编辑
            if st.session_state.get("tc_persona_name") != _tc_name:
                st.session_state.tc_persona_name = _tc_name
                st.session_state.tc_in = [c.get("input", "") for c in _tc]
                st.session_state.tc_exp = [c.get("expected_behavior", "") for c in _tc]
            st.divider()
            st.header("🧪 测试套件")
            st.caption(f"为「{_tc_name}」维护一组测试用例，一键跑出合格率。")
            if not st.session_state.tc_in:
                st.info("暂无测试用例，点下方「➕ 添加用例」开始。")
            for i in range(len(st.session_state.tc_in)):
                st.session_state.tc_in[i] = st.text_input(
                    f"用例 {i + 1} 输入", value=st.session_state.tc_in[i], key=f"tc_in_{i}")
                st.session_state.tc_exp[i] = st.text_area(
                    f"用例 {i + 1} 期望行为", value=st.session_state.tc_exp[i], key=f"tc_exp_{i}", height=70)
    
            _c1, _c2, _c3 = st.columns([1, 1, 1])
            with _c1:
                if st.button("➕ 添加用例", key="tc_add"):
                    st.session_state.tc_in.append("")
                    st.session_state.tc_exp.append("")
                    st.rerun()
            with _c2:
                if st.button("💾 保存测试用例", key="tc_save"):
                    _cases = [
                        {"input": a.strip(), "expected_behavior": b.strip()}
                        for a, b in zip(st.session_state.tc_in, st.session_state.tc_exp)
                        if a.strip()
                    ]
                    if set_test_cases(_tc_name, _cases):
                        st.success(f"已保存 {len(_cases)} 条测试用例")
                    else:
                        st.error("保存失败（人格不存在？）")
                    st.rerun()
            with _c3:
                if st.button("🏃 运行测试套件", key="tc_run"):
                    _cases = [
                        {"input": a.strip(), "expected_behavior": b.strip()}
                        for a, b in zip(st.session_state.tc_in, st.session_state.tc_exp)
                        if a.strip()
                    ]
                    if not _cases:
                        st.warning("请先添加至少一条测试用例")
                    else:
                        _pv = (get_persona_by_name(_tc_name, version=selected_version)
                               if selected_version is not None else temp_persona)
                        if not _pv:
                            st.error("找不到所选版本，请重新选择后再运行。")
                        else:
                            _prog = st.progress(0.0, text="准备中…")

                            def _sp(i, total, c):
                                _q = c.get("input") or c.get("question") or ""
                                _prog.progress(i / total, text=f"[{i}/{total}] {_q[:20] or '（空）'}")

                            with st.spinner(f"运行中（{len(_cases)} 条用例）…"):
                                _res = run_cases(_pv["persona"], _cases, st.session_state.get("custom_dims", []),
                                                 st.session_state.judge_model, progress_callback=_sp)
                            _prog.empty()
                            st.session_state.test_suite = _res
                            st.session_state.test_suite_name = _tc_name
                            st.rerun()
    
        # 一键回归测试（半自动化）：对【当前已加载人格】跑内置边缘场景题，自动对比上次
        st.divider()
        st.header("🚀 一键回归测试（边缘场景）")
        st.caption("对【当前已加载人格】跑一组内置边缘场景问题，自动对比上次结果。"
                   "改完提示词随手点一下即可，不必再手动维护测试用例。")
        _reg_qty = st.radio("题量", ["核心集（~15 题，推荐）", "完整 50 题（更慢）"],
                            key="reg_qty", horizontal=True)
        _reg_quick = st.checkbox("快速验证（只跑第 1 题，用于排查连通性）", value=False, key="reg_quick")
        if st.session_state.current_persona is None:
            st.info("请先在左侧「加载此人格」或生成/录入一个人格，再运行回归测试。")
        else:
            _reg_name = st.session_state.current_persona["name"]
            _reg_qs_full = FULL_QUESTIONS if _reg_qty.startswith("完整") else CORE_QUESTIONS
            _reg_qs = _reg_qs_full[:1] if _reg_quick else _reg_qs_full
            _est_min = (1 if _reg_quick else (3 if _reg_qty.startswith("完整") else 1))
            st.caption(f"预计跑 {len(_reg_qs)} 题 · {len(_reg_qs)} 条对话 + {len(_reg_qs) * 5} 次评分调用"
                       f"（{DEFAULT_WORKERS} 题并发、评分已并行，重复运行还会命中缓存）·"
                       f"约 {_est_min} 分钟（取决于网络/模型速度）。请耐心等待，避免重复点按钮。")
            if st.button("运行一键回归", key="oneclick_run"):
                _reg_pp = st.session_state.current_persona["persona"]
                _bpath = config.baseline_path(_reg_name)
                _rpath = config.regression_report_path(_reg_name)
    
                # 实时进度条 + 状态文本（关键：让用户知道没卡死）
                _progress = st.progress(0.0, text="准备中…")
                _status = st.empty()
    
                def _on_progress(idx, total, q):
                    _progress.progress(idx / total, text=f"[{idx}/{total}] {q['category']} | {q['question'][:20] or '（空）'}")
                    _status.caption(f"正在跑：{q['category']} — {q['question'][:30] or '（空）'}")
    
                _ok = True
                _err = None
                with st.spinner(f"回归计算中（{len(_reg_qs)} 题）…"):
                    try:
                        _res, _rep = run_regression_suite(
                            _reg_pp, _reg_qs, baseline_path=_bpath, report_path=_rpath,
                            mock=False, judge_model=st.session_state.judge_model,
                            progress_callback=_on_progress)
                        st.session_state.oneclick_regression = {
                            "md": _rep, "name": _reg_name, "path": _rpath}
                    except Exception as e:
                        _ok = False
                        _err = e
    
                # 收尾：清掉进度占位（让报告区域干净），把错误移出 spinner 上下文保证可见
                _progress.empty()
                _status.empty()
                if not _ok:
                    st.error(f"⚠️ 回归测试失败：**{type(_err).__name__}: {_err}**\n\n"
                             f"常见原因：① `.env` 没配 / API Key 无效；② 侧边栏「裁判模型」选了不存在的模型名；"
                             f"③ 网络不通 / API 限流 / 余额不足。\n\n"
                             f"建议先勾选「快速验证」只跑 1 题复现一次，确认是连接问题还是其他。")
                st.rerun()
    
        st.divider()
    st.header("🛠️ 生成新人格")
    new_desc = st.text_input("人格描述", placeholder="例如：一个喜欢恐龙的5岁女孩")
    gen_name = st.text_input("人格名称（可选）", placeholder="留空则按描述自动命名")
    
    # 模板列表复用脚本顶部已加载的 all_personas（新增/保存人格都会触发 rerun 重新加载）
    template_options = ["（不使用模板）"] + persona_names
    selected_template = st.selectbox("参考模板（可选）", template_options)
    
    strict_mode=st.checkbox("严格模仿参考模板风格",value=True)
    if st.button("生成人格"):
        if new_desc:
            with st.spinner("生成中..."):
                kwargs = {"description": new_desc, "strict": strict_mode}
                if selected_template != "（不使用模板）":
                    kwargs["template_name"] = selected_template
                persona_text = safe_tool(generate_persona, kwargs, "生成人格")
                if persona_text is None:
                    st.stop()
                from tools.persona_library import add_user_persona
                # 解决“生成人格同名覆盖”问题：有名字用名字，否则按描述自动命名
                final_name = gen_name.strip() if gen_name and gen_name.strip() else f"人格_{new_desc[:12]}"
                version_num=add_user_persona(final_name,persona_text,note=f"基于描述：{new_desc}")
                st.session_state.current_persona = {"name": final_name, "persona": persona_text}
                st.session_state.messages=[]
                save_last_active(st.session_state.current_persona)
                st.success(f"已生成人格：{final_name}")
                st.rerun()
        else:
            st.warning("请输入描述")
    st.divider()
    st.header("⚙️ 模式")
    comparison_mode = st.checkbox("对比模式（同时测试两个人格）")
    st.session_state.comparison_mode = comparison_mode
    if comparison_mode:
        st.info("对比模式：你将同时测试两个人格，并排显示结果。")
        persona_a_name = st.selectbox("人格 A", persona_names, key="a")
        persona_b_name = st.selectbox("人格 B", persona_names, key="b")
        if st.button("加载对比人格"):
            st.session_state.persona_a = get_persona_by_name(persona_a_name)
            st.session_state.persona_b = get_persona_by_name(persona_b_name)
            st.session_state.messages = []
            st.session_state.last_compare = None
            st.rerun()

    st.divider()
    st.header("📝录入人格")
    with st.form("add_persona_form"):
        new_name=st.text_input("人格名称",placeholder="例如：我的二次元男生")
        new_prompt=st.text_area("人格提示词",height=200,
                                placeholder="粘贴你之前测试通过的完整提示词...")
        submitted=st.form_submit_button("保存人格")
        if submitted:
            if new_name and new_prompt:
                from tools.persona_library import add_user_persona
                add_user_persona(new_name,new_prompt)
                st.success(f"人格「{new_name}」已保存！")
                st.rerun()
            else:
                st.warning("请填写完整信息")

    st.divider()
    st.header("📂 测试报告")
    try:
        report_files = sorted(
            [f for f in os.listdir(REPORTS_DIR) if f.endswith(".json")],
            reverse=True,
        )
    except FileNotFoundError:
        report_files = []
    if not report_files:
        st.caption("暂无保存的报告")
    for rf in report_files:
        _c1, _c2 = st.columns([3, 1])
        with _c1:
            st.caption(rf)
        with _c2:
            if st.button("🗑", key=f"del_{rf}"):
                os.remove(os.path.join(REPORTS_DIR, rf))
                st.rerun()

# 主区域
# 版本对比视图：优先于普通视图；点“关闭对比”返回
if st.session_state.version_diff:
    _render_version_diff(st.session_state.version_diff)
    if st.button("关闭对比", key="close_diff"):
        st.session_state.version_diff = None
        st.rerun()
elif st.session_state.comparison_mode and st.session_state.persona_a and st.session_state.persona_b:
    # 对比模式：两列
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"人格 A: {st.session_state.persona_a['name']}")
        st.text_area("提示词", st.session_state.persona_a["persona"], height=200, disabled=True)

    with col2:
        st.subheader(f"人格 B: {st.session_state.persona_b['name']}")
        st.text_area("提示词", st.session_state.persona_b["persona"], height=200, disabled=True)

    # 用户输入（共用一个输入框）
    user_input = st.chat_input("输入测试问题...")
    if user_input:
        # 人格 A 流式对话（流式失败则回退非流式 invoke）
        try:
            with st.spinner("人格 A 回答中..."):
                resp_a = st.write_stream(
                    stream_persona_response(st.session_state.persona_a["persona"], user_input, None)
                )
        except Exception as e_stream:
            try:
                resp_a = persona_chat.invoke({
                    "persona_prompt": st.session_state.persona_a["persona"],
                    "user_input": user_input,
                    "history": None,
                })
            except Exception as e_inv:
                st.error(f"⚠️ 人格 A 对话失败：{e_inv}（流式亦失败：{e_stream}）"); st.stop()
        # 人格 B 流式对话（流式失败则回退非流式 invoke）
        try:
            with st.spinner("人格 B 回答中..."):
                resp_b = st.write_stream(
                    stream_persona_response(st.session_state.persona_b["persona"], user_input, None)
                )
        except Exception as e_stream:
            try:
                resp_b = persona_chat.invoke({
                    "persona_prompt": st.session_state.persona_b["persona"],
                    "user_input": user_input,
                    "history": None,
                })
            except Exception as e_inv:
                st.error(f"⚠️ 人格 B 对话失败：{e_inv}（流式亦失败：{e_stream}）"); st.stop()

        # A/B 两侧评估并行（评估内部本身也已并发：标识∥基线 → 3 评委并行）。
        # 线程内不能调用 st.*（Streamlit UI 对象非线程安全），所以只在线程里拿结果/异常，
        # 回到主线程统一渲染错误提示。
        _cd = st.session_state.get("custom_dims", [])
        _jm = st.session_state.judge_model

        def _run_eval(persona_prompt, toy_response):
            try:
                return evaluate_persona.invoke({
                    "persona_prompt": persona_prompt,
                    "user_input": user_input,
                    "toy_response": toy_response,
                    "custom_dims": _cd,
                    "judge_model": _jm
                }), None
            except Exception as e:
                return None, e

        with ThreadPoolExecutor(max_workers=2) as _pool:
            _fa = _pool.submit(_run_eval, st.session_state.persona_a["persona"], resp_a)
            _fb = _pool.submit(_run_eval, st.session_state.persona_b["persona"], resp_b)
            eval_a, _err_a = _fa.result()
            eval_b, _err_b = _fb.result()
        if eval_a is None or eval_b is None:
            if _err_a:
                st.error(f"⚠️ 人格 A 评估失败：{_err_a}\n请检查网络或 API Key 后重试。")
            if _err_b:
                st.error(f"⚠️ 人格 B 评估失败：{_err_b}\n请检查网络或 API Key 后重试。")
            st.stop()
        # 暂存结果到 session_state：rerun 后从 session_state 展示，
        # 保证“保存/导出”按钮在后续交互中依然可用（避免整块随用户输入丢失）。
        st.session_state.last_compare = {
            "user_input": user_input,
            "resp_a": resp_a,
            "resp_b": resp_b,
            "eval_a": json.loads(eval_a),
            "eval_b": json.loads(eval_b),
        }
        st.rerun()

    # 展示最近一次对比结果（从 session_state 读取，可反复保存/导出）
    lc = st.session_state.get("last_compare")
    if lc:
        _ui = lc["user_input"]
        _ra, _rb = lc["resp_a"], lc["resp_b"]
        _ea, _eb = lc["eval_a"], lc["eval_b"]
        col1, col2 = st.columns(2)
        with col1:
            _lift_a = _ea.get("persona_lift", 0)
            _conf_a = _ea.get("confidence", "?")
            _conf_a_badge = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(_conf_a, "⚪")
            st.markdown(f"**用户：** {_ui}")
            st.markdown(f"**回答 A：** {_ra}")
            st.markdown(f"**评分：** {_ea['overall_score']}/10 · 基线 {_lift_a:+d} · 裁判 {_ea.get('judge_model', '?')} · 置信度 {_conf_a_badge}")
            st.markdown(f"**理由：** {_ea['reason']}")
        with col2:
            _lift_b = _eb.get("persona_lift", 0)
            _conf_b = _eb.get("confidence", "?")
            _conf_b_badge = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(_conf_b, "⚪")
            st.markdown(f"**回答 B：** {_rb}")
            st.markdown(f"**评分：** {_eb['overall_score']}/10 · 基线 {_lift_b:+d} · 裁判 {_eb.get('judge_model', '?')} · 置信度 {_conf_b_badge}")
            st.markdown(f"**理由：** {_eb['reason']}")

        # 保存报告
        if st.button("保存本次测试报告"):
            report = {
                "timestamp": datetime.now().isoformat(),
                "user_input": _ui,
                "persona_a": {
                    "name": st.session_state.persona_a["name"],
                    "response": _ra,
                    "evaluation": _ea
                },
                "persona_b": {
                    "name": st.session_state.persona_b["name"],
                    "response": _rb,
                    "evaluation": _eb
                }
            }
            os.makedirs("data/test_reports", exist_ok=True)
            filename = f"data/test_reports/compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            st.success(f"报告已保存至 {filename}")
            # 导出：Markdown / PDF
            _md = to_markdown(report)
            st.download_button("⬇️ 下载 Markdown", _md,
                               file_name=f"compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                               mime="text/markdown")
            _pdf_path = to_pdf(report)
            if _pdf_path and os.path.exists(_pdf_path):
                with open(_pdf_path, "rb") as _f:
                    st.download_button("⬇️ 下载 PDF", _f.read(),
                                       file_name=os.path.basename(_pdf_path),
                                       mime="application/pdf")

elif st.session_state.current_persona:
    # 单人格模式
    name = st.session_state.current_persona["name"]
    persona = st.session_state.current_persona["persona"]
    st.subheader(f"当前人格：{name}")
    st.text_area("人格提示词", persona, height=200, disabled=True)

    # 显示聊天历史
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 最新一次评分常驻显示（避免 rerun 后丢失）；维度动态展示（含自定义维度）
    if st.session_state.last_eval_data:
        ed = st.session_state.last_eval_data
        with st.container(border=True):
            # 顶部：综合分 + 置信度 + 评分波动
            _conf = ed.get("confidence", "?")
            _conf_badge = {"high": "🟢 高", "medium": "🟡 中", "low": "🔴 低"}.get(_conf, f"⚪ {_conf}")
            _range = ed.get("score_range", 0)
            st.markdown(f"**📊 综合评分：{ed.get('overall_score', '?')}/10** · 置信度 {_conf_badge} · 裁判 {ed.get('judge_model', '?')} · 3 次评分波动 ±{_range}")

            # 各维度
            _dim_md = "  \n".join(f"- **{_l}**：{_s}/10" for _l, _s in iter_dimensions(ed))
            st.markdown(_dim_md)

            # 基线对比
            _lift = ed.get("persona_lift", 0)
            _lift_str = f"+{_lift}" if _lift > 0 else str(_lift)
            st.markdown(f"**基线对比**（vs 普通助手）：`{_lift_str}` — 0=一样，正数=更有特色")

            # 证据命中 / 缺失
            _hit = ed.get("evidence_hit") or []
            _miss = ed.get("evidence_miss") or []
            if _hit:
                st.markdown("**✅ 证据命中**")
                for e in _hit:
                    st.markdown(f"  - {e}")
            if _miss:
                st.markdown("**❌ 证据缺失**")
                for e in _miss:
                    st.markdown(f"  - {e}")

            # 标识抽取失败提示（硬规则输入缺失时明确告知，不再静默）
            if ed.get("markers_extracted") is False:
                st.markdown("**⚠️ 标识抽取失败**：本次评估缺少硬规则输入（口头禅/身份事实未参与检查），"
                            "分数与证据可能偏差，建议重试。")

            # 硬规则检查
            hr = ed.get("hard_rules") or {}
            _hr_lines = []
            if hr.get("catches_hit"):
                _hr_lines.append(f"  - ✅ 口头禅命中：`{', '.join(hr['catches_hit'])}`")
            if hr.get("catches_miss"):
                _hr_lines.append(f"  - ❌ 口头禅缺失：`{', '.join(hr['catches_miss'])}`")
            if hr.get("facts_hit"):
                _hr_lines.append(f"  - ✅ 身份事实命中：`{', '.join(hr['facts_hit'])}`")
            if hr.get("facts_miss"):
                _hr_lines.append(f"  - ❌ 身份事实缺失：`{', '.join(hr['facts_miss'])}`")
            if _hr_lines:
                st.markdown("**🔍 硬规则检查**")
                for line in _hr_lines:
                    st.markdown(line)

            # 理由 + 建议
            st.markdown(f"**理由**：{ed.get('reason', '')}")
            st.markdown(f"**建议**：{ed.get('suggestion', '')}")

    # 用户输入（流式对话）
    if prompt := st.chat_input("输入测试问题..."):
        # 显示用户消息
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 流式回答（history 不含当前这条用户消息，避免重复拼接）
        # 若流式失败（如 deepseek-reasoner 不支持 stream=True、网络抖动），回退到非流式 invoke，保证一定有回答
        with st.chat_message("assistant"):
            try:
                response = st.write_stream(
                    stream_persona_response(persona, prompt, st.session_state.messages[:-1])
                )
            except Exception as e_stream:
                try:
                    response = persona_chat.invoke({
                        "persona_prompt": persona,
                        "user_input": prompt,
                        "history": st.session_state.messages[:-1],
                    })
                    st.write(response)
                except Exception as e_inv:
                    st.error(f"⚠️ 人格对话失败：{e_inv}（流式亦失败：{e_stream}）")
                    st.stop()
        st.session_state.messages.append({"role": "assistant", "content": response})
        save_chat_history(name, st.session_state.messages)

        # 评估
        with st.spinner("评估中..."):
            evaluation = safe_tool(evaluate_persona, {
                "persona_prompt": persona,
                "user_input": prompt,
                "toy_response": response,
                "custom_dims": st.session_state.get("custom_dims", []),
                "judge_model": st.session_state.judge_model
            }, "评估")
        if evaluation is None:
            st.stop()
        st.session_state.last_evaluation = evaluation
        st.session_state.last_eval_data = json.loads(evaluation)
        st.session_state.last_user_input = prompt
        st.rerun()

    # —— 优化人格：先收集用户反馈，再针对性优化 ——
    # 关键：按钮必须放在 if prompt 块【之外】，否则点击后 rerun 时整块不执行，按钮“没反应”
    if st.session_state.await_feedback:
        st.divider()
        st.subheader("✏️ 这个回答哪里不对？")
        # 评委建议一键采用：省掉「测试→发现弱点→优化」之间的人工誊抄
        _sug = (st.session_state.last_eval_data or {}).get("suggestion")
        _sug = _sug.strip() if isinstance(_sug, str) else ""
        if _sug and _sug not in ("（无）", "（mock）"):
            st.info(f"💡 评委建议：{_sug}")
            if st.button("⚡ 采用评估建议（填入下方反馈框，可再编辑）", key="fb_use_suggestion"):
                st.session_state.fb_input = _sug
                st.rerun()
        fb = st.text_area(
            "请描述你期望的回答 / 哪里不满意（越具体越好）",
            key="fb_input", height=120,
            placeholder="例如：它不该用成年人的语气，应该更奶声奶气一点"
        )
        _c_ok, _c_cancel = st.columns([1, 1])
        with _c_ok:
            if st.button("提交并优化人格", key="fb_submit"):
                if fb.strip():
                    with st.spinner("针对性优化中..."):
                        new_persona = safe_tool(optimize_persona, {
                            "persona_prompt": persona,
                            "user_input": st.session_state.last_user_input,
                            "toy_response": _last_assistant(st.session_state.messages),
                            "evaluation": st.session_state.last_evaluation,
                            "feedback": fb.strip()
                        }, "优化人格")
                    if new_persona is None:
                        st.stop()
                    from tools.persona_library import add_user_persona
                    version_num = add_user_persona(name, new_persona, note=f"用户反馈优化：{fb.strip()[:40]}")
                    st.session_state.current_persona = get_persona_by_name(name)  # 重新加载最新版本
                    st.session_state.messages = []
                    save_chat_history(name, [])
                    save_last_active(st.session_state.current_persona)
                    st.session_state.await_feedback = False
                    st.session_state.last_eval_data = None
                    st.success(f"✅ 已根据你反馈优化并保存为新版本 v{version_num}")
                    st.rerun()
                else:
                    st.warning("请先写下你的反馈再提交")
        with _c_cancel:
            if st.button("取消", key="fb_cancel"):
                st.session_state.await_feedback = False
                st.rerun()
    else:
        if st.button("这个回答不符合预期，优化人格"):
            st.session_state.await_feedback = True
            st.rerun()

    # 保存本次测试记录（还没有评估结果时拒绝保存，避免生成空报告）
    if st.button("保存本次测试"):
        if st.session_state.last_eval_data is None:
            st.warning("本次会话还没有评估结果，先发送一条测试问题完成评估，再保存报告。")
        else:
            report = {
                "timestamp": datetime.now().isoformat(),
                "persona_name": name,
                "persona_prompt": persona,
                "user_input": st.session_state.last_user_input,
                "response": _last_assistant(st.session_state.messages),
                "evaluation": st.session_state.last_eval_data
            }
            os.makedirs(REPORTS_DIR, exist_ok=True)
            filename = f"{REPORTS_DIR}/test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            st.success(f"报告已保存至 {filename}")
            # 导出：Markdown / PDF
            _md = to_markdown(report)
            st.download_button("⬇️ 下载 Markdown", _md,
                               file_name=f"{name}_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                               mime="text/markdown")
            _pdf_path = to_pdf(report)
            if _pdf_path and os.path.exists(_pdf_path):
                with open(_pdf_path, "rb") as _f:
                    st.download_button("⬇️ 下载 PDF", _f.read(),
                                       file_name=os.path.basename(_pdf_path),
                                       mime="application/pdf")

    # 测试套件结果展示
    if st.session_state.get("test_suite") is not None:
        ts = st.session_state.test_suite
        ts_name = st.session_state.get("test_suite_name", "")
        st.divider()
        st.subheader(f"🧪 测试套件结果：{ts_name}")
        _total = len(ts)
        _failed = [r for r in ts if r.get("error")]
        _valid = [r for r in ts if row_passed(r) is not None]
        _passed = sum(1 for r in _valid if row_passed(r))
        _pct = round(_passed / len(_valid) * 100) if _valid else 0
        _na_note = "（无期望行为的用例不参与合格率统计）" if (_total - len(_valid) - len(_failed)) else ""
        # 失败横幅：避免「全部失败」被误读成「合格率 0/0」
        if _failed:
            st.error(f"⚠️ {len(_failed)}/{_total} 条用例运行失败（见下方各用例展开项）。"
                     f"通常原因：API Key 未生效 / 网络不通 / 余额不足 / 模型名错误。"
                     f"请检查凭据后重试，失败用例不计入合格率。")
        st.markdown(f"**合格率：{_passed}/{len(_valid)}（{_pct}%）** · 共 {_total} 条用例"
                    f" · 失败 {len(_failed)} 条{_na_note}")
        for i, r in enumerate(ts):
            _sc = row_score(r)
            _pd = row_passed(r)
            if r.get("error"):
                _badge = "💥"
            else:
                _badge = "✅" if _pd is True else ("⚪" if _pd is None else "❌")
            with st.expander(f"用例 {i + 1}: {r['input'][:40]}  {_badge}  "
                             f"({('-' if _sc is None else _sc)}/10)"):
                st.markdown(f"**输入**：{r['input']}")
                if r.get("expected"):
                    st.markdown(f"**期望行为**：{r['expected']}")
                if r.get("error"):
                    st.error(f"**运行失败**：{r['error']}")
                    continue
                st.markdown(f"**实际回答**：{r['response']}")
                st.markdown(f"**综合评分**：{_sc}/10")
                _miss = r["eval"].get("evidence_miss") or []
                if _miss:
                    st.markdown("**❌ 证据缺失**")
                    for e in _miss:
                        st.markdown(f"- {e}")
                _sug = r["eval"].get("suggestion")
                if _sug:
                    st.markdown(f"**建议**：{_sug}")
        if st.button("清除测试套件结果", key="clear_ts"):
            st.session_state.test_suite = None
            st.session_state.test_suite_name = ""
            st.rerun()

    # 版本回归结果展示
    if st.session_state.get("version_regression") is not None:
        vr = st.session_state.version_regression
        st.divider()
        st.subheader(f"🔁 版本回归：{vr['name']}")
        st.markdown(f"平均得分：vA **{vr['avg_a']}** → vB **{vr['avg_b']}**"
                    f"  ·  合格率：{vr['pass_a']}% → {vr['pass_b']}%")
        if vr.get("failed_a") or vr.get("failed_b"):
            st.error(f"⚠️ 运行失败：vA {vr['failed_a']} 条、vB {vr['failed_b']} 条"
                     f"（API Key / 网络 / 余额问题）。失败用例不计入平均分与合格率。")
        _rows = []
        for r in vr["rows"]:
            _rows.append({
                "输入": r["input"],
                "vA 得分": r["score_a"] if r["score_a"] is not None else "-",
                "vB 得分": r["score_b"] if r["score_b"] is not None else "-",
                "Δ": (f"{r['delta']:+d}" if r["delta"] is not None else "-"),
                "vA 通过": "✅" if r["pass_a"] is True else ("⚪" if r["pass_a"] is None else "❌"),
                "vB 通过": "✅" if r["pass_b"] is True else ("⚪" if r["pass_b"] is None else "❌"),
            })
        st.dataframe(_rows, width="stretch")
        if st.button("清除回归结果", key="clear_vr"):
            st.session_state.version_regression = None
            st.rerun()

    # 一键回归测试结果展示（半自动化：对当前已加载人格的内置边缘场景体检）
    if st.session_state.get("oneclick_regression") is not None:
        ocr = st.session_state.oneclick_regression
        st.divider()
        st.subheader(f"🚀 一键回归测试结果：{ocr['name']}")
        st.caption("内置边缘场景问题（身份挑战 / 越界诱导 / 角色混淆 / 空输入 / 多轮一致 / 事实边界 / 冲突指令 / 情感操纵 / 能力外）。"
                   "表格中的「波动」「趋势」对比的是上一次运行结果。")
        st.markdown(ocr["md"])
        try:
            with open(ocr["path"], "r", encoding="utf-8") as _f:
                _ocr_md = _f.read()
            st.download_button("⬇️ 下载评测报告（Markdown）", _ocr_md,
                               file_name=f"regression_{ocr['name']}.md", mime="text/markdown")
        except Exception:
            pass
        if st.button("清除回归结果", key="clear_ocr"):
            st.session_state.oneclick_regression = None
            st.rerun()

else:
    st.info("从左侧选择一个人格，或生成新人格开始测试。")

