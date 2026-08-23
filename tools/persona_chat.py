from typing import Optional
from langchain_core.tools import tool
from tools.llm_config import get_llm

def _build_prompt(persona_prompt: str, user_input: str, history: list = None) -> str:
    """拼接「系统人设 + 历史 + 当前输入」，得到发给模型的完整 prompt。"""
    conversation = []
    if history:
        for msg in history:
            role_label = "用户" if msg.get("role") == "user" else "玩具"
            conversation.append(f"{role_label}：{msg.get('content', '')}")
    conversation.append(f"用户：{user_input}")
    conversation.append("玩具：")
    return f"{persona_prompt}\n\n" + "\n".join(conversation)


@tool
def persona_chat(persona_prompt: str, user_input: str, history: Optional[list] = None) -> str:
    """
    用人格提示词与用户对话，返回玩具的回答（非流式，供 Agent / 测试使用）。
    参数：
        persona_prompt: 人格设定文本
        user_input: 用户当前的输入
        history: 可选，历史对话列表，每项形如 {"role": "user"/"assistant", "content": "..."}
    返回：
        玩具的回答（字符串）
    """
    response = get_llm().invoke(_build_prompt(persona_prompt, user_input, history))
    return response.content


def stream_persona_response(persona_prompt: str, user_input: str, history: list = None):
    """流式版本：逐块 yield 玩具回答文本，供 Streamlit 的 st.write_stream 使用。"""
    for chunk in get_llm().stream(_build_prompt(persona_prompt, user_input, history)):
        if chunk.content:
            yield chunk.content

if __name__ == "__main__":
    test_persona = "你是一个5岁的小女孩，名叫小美。你喜欢公主、恐龙和冰淇淋。说话奶声奶气，经常用叠词。"
    test_input = "你好，你叫什么名字？"
    answer = persona_chat.invoke({
        "persona_prompt": test_persona,
        "user_input": test_input
    })
    print(f"用户：{test_input}")
    print(f"玩具：{answer}")