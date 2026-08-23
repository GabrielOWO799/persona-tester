# tools/persona_generator.py
import json
import os
from langchain_core.tools import tool
from tools.llm_config import get_llm
from tools.persona_library import get_persona_by_name, get_all_personas

@tool
def generate_persona(description: str, template_name: str = None, strict: bool = True) -> str:
    """
    根据简短描述生成详细的人格提示词。可指定一个已有的人格作为参考模板。
    参数：
        description: 简短描述，例如“一个喜欢二次元的男生”
        template_name: 可选，已有的人格名称，将作为生成模板（示例）
        strict: 是否严格模仿模板风格。True 时把模板作为强约束注入；False 时仅作为参考示例。
    返回：
        详细的人格设定文本
    """
    template = get_persona_by_name(template_name) if template_name else None

    # 严格模仿：把模板作为强约束直接注入，要求不得偏离其风格基调
    if template_name and template and strict:
        template_block = (
            "【严格模仿模板】你必须严格沿用以下人格的语气、语言风格、口头禅与身份设定，"
            "不得擅自改变其性格基调；仅根据用户描述补充贴合的细节：\n"
            f"{template['name']}\n{template['persona']}"
        )
        prompt = f"""
你是一个专业的人格设定专家。请根据以下描述，生成一个适合AI玩具的详细人格提示词。

要求：
- 使用第二人称“你”来写
- 必须包含：年龄、名字、性格特点、常用口头禅、兴趣爱好、至少一个典型对话示例
- 语言生动具体，便于模型扮演

{template_block}

现在，请根据以下描述生成人格（严格沿用上面模板的风格基调，不得偏离）：
描述：{description}

输出：
"""
    else:
        # 非严格：模板仅作参考示例；无模板则用内置示例
        if template_name and template:
            examples = [f"参考示例：{template['name']}\n{template['persona']}"]
        else:
            examples = [
                "参考示例1（热血少年）：\n你是15岁的高中生，名叫小枫，最喜欢看《火影忍者》。你性格热血，遇到困难会大喊“我一定会赢！”。你的口头禅是“说到做到，这就是我的忍道！”。",
                "参考示例2（温柔学姐）：\n你是18岁的大学生，名叫林萱，喜欢阅读和烘焙。你说话轻声细语，总是带着微笑，常用“没关系”、“慢慢来”鼓励别人。你的口头禅是“你做得很好哦”。"
            ]
        prompt = f"""
你是一个专业的人格设定专家。请根据以下描述，生成一个适合AI玩具的详细人格提示词。

要求：
- 使用第二人称“你”来写
- 必须包含：年龄、名字、性格特点、常用口头禅、兴趣爱好、至少一个典型对话示例
- 语言生动具体，便于模型扮演

{"".join(examples)}

现在，请根据以下描述生成人格：
描述：{description}

输出：
"""
    response = get_llm().invoke(prompt)
    return response.content