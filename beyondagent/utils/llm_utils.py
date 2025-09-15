import os
from openai import OpenAI

def call_qwen(message, model="qwen-max"):
    """
    调用 Qwen 模型进行对话。

    参数:
        user_message (str): 用户输入的问题。
        system_message (str): 系统角色的提示词。
        model (str): 使用的模型名称（默认 qwen-max）。

    返回:
        str: 模型的回复内容。
    """
    try:
        client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        completion = client.chat.completions.create(
            model=model,
            messages=message,
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"错误信息：{e}")
        return None
