# tools/llm_config.py
"""全局 LLM 配置：让模型与温度可在界面上动态调整，而不再写死。"""
import os
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek

load_dotenv()

# 全局可变的模型与温度（由 app.py 侧边栏在运行时设置）
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
TEMPERATURE = 0.7

# 重试配置：调用 DeepSeek 遇限流 / 超时等临时错误时自动指数退避重试
RETRY_ATTEMPTS = 3


def set_model(model: str):
    """切换模型（如 deepseek-chat / deepseek-reasoner / 自定义名）。"""
    global MODEL
    MODEL = model or "deepseek-chat"


def set_temperature(t: float):
    """设置全局温度（0~1）。"""
    global TEMPERATURE
    TEMPERATURE = float(t)


# 客户端缓存：按 (model, temperature) 复用 ChatDeepSeek 实例。
# 评估器会反复调用 get_llm(0) / get_llm(0.4) / get_llm(0, model=judge_model) 等，
# 缓存可避免每次都新建对象（单实例开销很小，但在高频评估循环里更稳、更省）。
_LLM_CACHE: dict = {}


def get_llm(temperature: float = None, model: str = None, retry: bool = True) -> ChatDeepSeek:
    """
    返回 ChatDeepSeek 实例（按 model+temperature 复用，避免重复构造）。
    temperature 为 None 时使用全局 TEMPERATURE；model 为 None 时使用全局 MODEL。
    model 传具体值（如 'deepseek-reasoner'）可让裁判与演员用不同模型，实现解耦。
    retry=True 时叠加 tenacity 指数退避重试，应对限流 / 超时等临时错误，提升评估稳定性。
    """
    _model = model if model else MODEL
    _temp = temperature if temperature is not None else TEMPERATURE
    # 统一成可哈希的稳定 key：int 0 与 float 0.0 视为同一组合，避免重复建实例
    key = (str(_model), float(_temp), bool(retry))
    if key not in _LLM_CACHE:
        # deepseek-reasoner 等推理模型不支持 stream=True，按模型名关闭流式，
        # 避免每次调用先失败一次再靠上层 try/except 回退（白付一次往返）
        _streaming = "reasoner" not in _model.lower()
        llm = ChatDeepSeek(
            model=_model,
            temperature=_temp,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            streaming=_streaming,
        )
        if retry:
            # with_retry 内部使用 tenacity 做指数退避重试：遇限流 / 超时等临时错误自动重试，
            # 不重试鉴权 / 参数类错误；让偶发网络抖动不再直接白屏，评估更稳。
            llm = llm.with_retry(
                stop_after_attempt=RETRY_ATTEMPTS,
                wait_exponential_jitter=True,
            )
        _LLM_CACHE[key] = llm
    return _LLM_CACHE[key]
