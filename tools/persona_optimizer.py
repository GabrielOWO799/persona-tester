# tools/persona_optimizer.py
import json
import os
from langchain_core.tools import tool
from tools.llm_config import get_llm

@tool
def optimize_persona(persona_prompt: str, user_input: str, toy_response: str, evaluation: str, feedback: str = None) -> str:
    """
    根据用户反馈的评估结果，改进人格提示词，使其在后续对话中更符合预期。
    参数：
        persona_prompt: 当前人格提示词
        user_input: 用户输入
        toy_response: 玩具的实际回答
        evaluation: 评估结果 JSON 字符串
        feedback: 可选，用户的具体反馈（期望怎么答 / 哪里不满意），用于针对性优化
    返回：
        优化后的人格提示词
    """
    # 解析评估结果（如果是字典则直接使用，否则尝试加载）
    try:
        eval_data = json.loads(evaluation) if isinstance(evaluation, str) else evaluation
    except:
        eval_data = {"suggestion": "回答不够理想"}

    # 用户具体反馈：拼进提示词，让优化更有针对性
    feedback_block = ""
    if feedback and feedback.strip():
        feedback_block = f"\n用户的具体反馈：{feedback.strip()}\n请重点针对以上反馈进行调整，使未来回答更符合用户的期望。"

    prompt = f"""
你是一个人格设定专家。用户对当前人格的某个回答不满意，请根据反馈改进人格提示词。

当前人格：
{persona_prompt}

用户问：{user_input}
玩具回答：{toy_response}
评估反馈：{eval_data.get('suggestion', '需要改进')}
{feedback_block}

请生成一个改进后的人格提示词，要求：
- 保留原有好的特质
- 针对反馈调整，使未来回答更符合设定
- 使用第二人称“你”来写
- 仍包含年龄、名字、性格、口头禅、爱好等要素

输出改进后的人格：
"""
    response = get_llm().invoke(prompt)
    return response.content