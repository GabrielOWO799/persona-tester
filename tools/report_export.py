# tools/report_export.py
"""把测试报告导出为 Markdown 与 PDF（含中文）。

兼容两种报告结构：
  - 单人格测试：{timestamp, persona_name, persona_prompt, user_input, response, evaluation}
  - 对比测试：  {timestamp, user_input, persona_a:{name,response,evaluation}, persona_b:{...}}
"""
import os
from datetime import datetime

import config

# 固定评估维度的中文标签（与 persona_evaluator.FIXED_DIMS 对应）
_FIXED_LABELS = {
    "consistency_score": "人格一致性",
    "style_score": "语言风格",
    "identity_score": "角色身份",
}


def iter_dimensions(eval_data):
    """从评估字典里取出各维度 (标签, 分数)，按出现顺序返回。

    维度分数字段的命名规则是 `<名称>_score`（如 consistency_score / 幽默感_score），
    而 overall_score / reason / suggestion 不是维度，需要排除。
    """
    dims = []
    if not eval_data or not isinstance(eval_data, dict):
        return dims
    for k, v in eval_data.items():
        if k.endswith("_score") and k != "overall_score":
            label = _FIXED_LABELS.get(k, k[:-6])  # 自定义维度：去掉尾缀 _score
            try:
                score = int(v)
            except Exception:
                score = v
            dims.append((label, score))
    return dims


def _lint_int(v):
    """容忍 int / float / str 的整数化（用于基线 lift 等可正可负的字段）。"""
    try:
        return int(round(float(v)))
    except Exception:
        return 0


def _eval_block_md(name, eval_data):
    parts = [f"### 评估结果：{name}"]
    if eval_data:
        head = f"- **综合评分**：{eval_data.get('overall_score', '?')}/10"
        conf = eval_data.get("confidence")
        rng = eval_data.get("score_range")
        if conf and rng is not None:
            head += f"（置信度 {conf}，3 次评分波动 ±{rng}）"
        parts.append(head)
        for label, score in iter_dimensions(eval_data):
            parts.append(f"- **{label}**：{score}/10")
        lift = eval_data.get("persona_lift")
        if lift is not None:
            parts.append(f"- **基线对比**（vs 普通助手）：{_lint_int(lift):+d}")
        for e in eval_data.get("evidence_hit") or []:
            parts.append(f"- ✅ 证据命中：{e}")
        for e in eval_data.get("evidence_miss") or []:
            parts.append(f"- ❌ 证据缺失：{e}")
        hr = eval_data.get("hard_rules") or {}
        if hr.get("catches_hit"):
            parts.append(f"- ✅ 口头禅命中：`{', '.join(hr['catches_hit'])}`")
        if hr.get("catches_miss"):
            parts.append(f"- ❌ 口头禅缺失：`{', '.join(hr['catches_miss'])}`")
        if hr.get("facts_hit"):
            parts.append(f"- ✅ 身份事实命中：`{', '.join(hr['facts_hit'])}`")
        if hr.get("facts_miss"):
            parts.append(f"- ❌ 身份事实缺失：`{', '.join(hr['facts_miss'])}`")
        if eval_data.get("markers_extracted") is False:
            parts.append("- ⚠️ 标识抽取失败，硬规则未参与本次检查")
        if eval_data.get("reason"):
            parts.append(f"- **理由**：{eval_data.get('reason')}")
        if eval_data.get("suggestion"):
            parts.append(f"- **建议**：{eval_data.get('suggestion')}")
    else:
        parts.append("（无评估数据）")
    return "\n".join(parts)


def to_markdown(report):
    """把报告 dict 转成 Markdown 文本（兼容单人格 / 对比两种结构）。"""
    lines = ["# 人格测试报告", ""]
    lines.append(f"**生成时间**：{report.get('timestamp', '')}")
    ui = report.get("user_input")
    if ui:
        lines.append(f"**测试问题**：{ui}")
    lines.append("")

    pa = report.get("persona_a")
    pb = report.get("persona_b")
    if pa and pb:
        # 对比报告
        lines.append(f"## 人格 A：{pa.get('name', '')}")
        prompt_a = pa.get("persona", pa.get("persona_prompt", ""))
        if prompt_a:
            lines.append(f"**提示词**：\n```\n{prompt_a}\n```")
        lines.append(f"**回答**：\n{pa.get('response', '')}")
        lines.append(_eval_block_md(pa.get("name", "A"), pa.get("evaluation")))
        lines.append("")
        lines.append(f"## 人格 B：{pb.get('name', '')}")
        prompt_b = pb.get("persona", pb.get("persona_prompt", ""))
        if prompt_b:
            lines.append(f"**提示词**：\n```\n{prompt_b}\n```")
        lines.append(f"**回答**：\n{pb.get('response', '')}")
        lines.append(_eval_block_md(pb.get("name", "B"), pb.get("evaluation")))
    else:
        name = report.get("persona_name", "")
        lines.append(f"## 人格：{name}")
        pp = report.get("persona_prompt", "")
        if pp:
            lines.append(f"**提示词**：\n```\n{pp}\n```")
        lines.append(f"**回答**：\n{report.get('response', '')}")
        lines.append(_eval_block_md(name, report.get("evaluation")))
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# PDF 导出（reportlab，使用内置中文字体 STSong-Light，无需额外字体文件）
# ----------------------------------------------------------------------------
def _esc(text):
    """转义 XML 特殊字符，并把换行转成 <br/>，供 Paragraph 使用。"""
    if text is None:
        text = ""
    s = str(text)
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    s = s.replace("\n", "<br/>")
    return s


_PDF_FONT = "STSong-Light"


def _eval_flowables(eval_data, styles):
    from reportlab.platypus import Paragraph, Spacer
    flow = []
    if eval_data:
        head = f"综合评分：{eval_data.get('overall_score', '?')}/10"
        conf = eval_data.get("confidence")
        rng = eval_data.get("score_range")
        if conf and rng is not None:
            head += f"（置信度 {conf}，3 次评分波动 ±{rng}）"
        flow.append(Paragraph(head, styles["zhBody"]))
        for label, score in iter_dimensions(eval_data):
            flow.append(Paragraph(f"{label}：{score}/10", styles["zhBody"]))
        lift = eval_data.get("persona_lift")
        if lift is not None:
            flow.append(Paragraph(f"基线对比（vs 普通助手）：{_lint_int(lift):+d}", styles["zhBody"]))
        for e in eval_data.get("evidence_hit") or []:
            flow.append(Paragraph(f"✅ 证据命中：{_esc(e)}", styles["zhBody"]))
        for e in eval_data.get("evidence_miss") or []:
            flow.append(Paragraph(f"❌ 证据缺失：{_esc(e)}", styles["zhBody"]))
        hr = eval_data.get("hard_rules") or {}
        if hr.get("catches_hit"):
            flow.append(Paragraph(f"✅ 口头禅命中：{_esc(', '.join(hr['catches_hit']))}", styles["zhBody"]))
        if hr.get("catches_miss"):
            flow.append(Paragraph(f"❌ 口头禅缺失：{_esc(', '.join(hr['catches_miss']))}", styles["zhBody"]))
        if hr.get("facts_hit"):
            flow.append(Paragraph(f"✅ 身份事实命中：{_esc(', '.join(hr['facts_hit']))}", styles["zhBody"]))
        if hr.get("facts_miss"):
            flow.append(Paragraph(f"❌ 身份事实缺失：{_esc(', '.join(hr['facts_miss']))}", styles["zhBody"]))
        if eval_data.get("markers_extracted") is False:
            flow.append(Paragraph("⚠️ 标识抽取失败，硬规则未参与本次检查", styles["zhBody"]))
        if eval_data.get("reason"):
            flow.append(Paragraph(f"理由：{_esc(eval_data.get('reason'))}", styles["zhBody"]))
        if eval_data.get("suggestion"):
            flow.append(Paragraph(f"建议：{_esc(eval_data.get('suggestion'))}", styles["zhBody"]))
    else:
        flow.append(Paragraph("（无评估数据）", styles["zhBody"]))
    flow.append(Spacer(1, 6))
    return flow


def to_pdf(report, path=None):
    """导出为 PDF，返回文件路径。path 为 None 时自动生成到 data/test_reports。"""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

    # 注册内置中文 CID 字体（无需外部 .ttf）
    global _PDF_FONT
    try:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        _PDF_FONT = "STSong-Light"
    except Exception:
        _PDF_FONT = "Helvetica"

    if path is None:
        os.makedirs(config.REPORTS_DIR, exist_ok=True)
        path = os.path.join(config.REPORTS_DIR, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")

    doc = SimpleDocTemplate(
        path, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("zhTitle", parent=styles["Title"], fontName=_PDF_FONT, fontSize=18))
    styles.add(ParagraphStyle("zhH2", parent=styles["Heading2"], fontName=_PDF_FONT, fontSize=13))
    styles.add(ParagraphStyle("zhBody", parent=styles["BodyText"], fontName=_PDF_FONT, fontSize=10, leading=15))
    styles.add(ParagraphStyle("zhMono", parent=styles["BodyText"], fontName=_PDF_FONT, fontSize=9, leading=13))

    story = []
    story.append(Paragraph("人格测试报告", styles["zhTitle"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"生成时间：{_esc(report.get('timestamp', ''))}", styles["zhBody"]))
    ui = report.get("user_input")
    if ui:
        story.append(Paragraph(f"测试问题：{_esc(ui)}", styles["zhBody"]))
    story.append(Spacer(1, 8))

    pa = report.get("persona_a")
    pb = report.get("persona_b")
    if pa and pb:
        story.append(Paragraph(f"人格 A：{_esc(pa.get('name', ''))}", styles["zhH2"]))
        story.append(Paragraph(_esc(pa.get("response", "")), styles["zhBody"]))
        story.extend(_eval_flowables(pa.get("evaluation"), styles))
        story.append(Paragraph(f"人格 B：{_esc(pb.get('name', ''))}", styles["zhH2"]))
        story.append(Paragraph(_esc(pb.get("response", "")), styles["zhBody"]))
        story.extend(_eval_flowables(pb.get("evaluation"), styles))
    else:
        name = report.get("persona_name", "")
        story.append(Paragraph(f"人格：{_esc(name)}", styles["zhH2"]))
        pp = report.get("persona_prompt", "")
        if pp:
            story.append(Paragraph("提示词：", styles["zhBody"]))
            story.append(Paragraph(_esc(pp), styles["zhMono"]))
        story.append(Paragraph(_esc(report.get("response", "")), styles["zhBody"]))
        story.extend(_eval_flowables(report.get("evaluation"), styles))

    doc.build(story)
    return path
