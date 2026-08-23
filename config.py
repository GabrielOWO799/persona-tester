# config.py
"""集中管理所有运行时路径。

全部路径锚定到项目根目录（本文件所在目录），无论从哪个工作目录启动
（streamlit run app.py / python test_suite.py / pytest）都读写同一位置，
替代此前散落在 app.py、test_suite.py、tools/* 里的相对路径硬编码。
"""
import os
import re

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REPORTS_DIR = os.path.join(DATA_DIR, "test_reports")          # 单次测试 / 对比测试报告
CHAT_HISTORY_DIR = os.path.join(DATA_DIR, "chat_history")     # 按人格 md5 存的聊天历史
REGRESSION_DIR = os.path.join(DATA_DIR, "regression")         # 一键回归的基线与报告

REFERENCE_PERSONAS_PATH = os.path.join(DATA_DIR, "reference_personas.json")
USER_PERSONAS_PATH = os.path.join(DATA_DIR, "user_personas.json")
LAST_ACTIVE_PATH = os.path.join(DATA_DIR, "last_active.json")


def _safe_name(name: str) -> str:
    """人格名可能含中文/特殊字符，清洗成安全文件名（保留中文）。"""
    return re.sub(r"[^\w\u4e00-\u9fff-]", "_", name or "default")


def baseline_path(name: str) -> str:
    """一键回归的基线文件路径（按人格名区分，避免不同人格互相污染对比）。"""
    return os.path.join(REGRESSION_DIR, f"baseline_{_safe_name(name)}.json")


def regression_report_path(name: str) -> str:
    """一键回归的 Markdown 报告路径。"""
    return os.path.join(REGRESSION_DIR, f"report_{_safe_name(name)}.md")


def ensure_dirs():
    """创建全部运行时目录。若历史上被误建为同名文件（如 test_reports 曾是文件）则先删除。"""
    for d in (DATA_DIR, REPORTS_DIR, CHAT_HISTORY_DIR, REGRESSION_DIR):
        if os.path.exists(d) and not os.path.isdir(d):
            os.remove(d)
        os.makedirs(d, exist_ok=True)
