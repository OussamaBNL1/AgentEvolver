from typing import Any, List, Dict
import torch
from loguru import logger

# apply chat_template to a message, and then convert back to message
def convert_tool_to_user_message(tool_message, format="qwen"):
    assert format == "qwen"

    if tool_message["role"] == "user":
        return tool_message
    elif tool_message["role"] == "tool" and len(tool_message["tool_calls"])>0:
        assert len(tool_message["tool_calls"])==1
        return {
            "role": "user",
            "content": str(tool_message["tool_calls"][0]['result'])
        }


def clip_state_content_correctly(tokenizer, state_content: str, max_env_len: int) -> str:
    """
    正确地截断state_content，确保不会破坏token边界
    
    Args:
        tokenizer: 分词器
        state_content: 要截断的内容
        max_env_len: 最大允许的token长度
    
    Returns:
        截断后的内容字符串
    """
    # 先tokenize检查长度
    tokens = tokenizer(state_content, return_tensors="pt", padding=False)["input_ids"][0]
    
    if len(tokens) <= max_env_len:
        return state_content
    
    # 如果超长，截断到max_env_len长度的token
    truncated_tokens = tokens[:max_env_len]
    
    # 更安全的方式：使用tokenizer的内置方法
    # 大多数tokenizer都有更好的处理方式
    if hasattr(tokenizer, 'decode'):
        # 首先尝试保留special tokens
        try:
            truncated_content = tokenizer.decode(truncated_tokens, skip_special_tokens=False)
            return truncated_content
        except:
            # 如果失败，可能是截断位置不当，尝试移除special tokens
            try:
                truncated_content = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
                return truncated_content
            except:
                # 最后的fallback：手动处理
                pass
    
    # 如果所有decode方法都失败，使用更保守的方法
    # 逐步减少token数量直到成功decode
    for i in range(min(10, max_env_len)):  # 最多尝试10次
        try:
            test_tokens = tokens[:max_env_len - i]
            truncated_content = tokenizer.decode(test_tokens, skip_special_tokens=False)
            logger.warning(f"Had to reduce token count by {i} to successfully decode")
            return truncated_content
        except:
            continue
    
    # 最终fallback：使用原始的字符截断方法
    logger.error("All token-based truncation methods failed, falling back to character truncation")
    return state_content[:max_env_len]


def get_batched_exponential_decay_weights_vectorized(
    lens: list[int],
    start_val: float = 10.0,
    end_val: float = 1.0,
    decay_reach_percent: float = 0.85,
    padding_value: float = 0.0,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """
    高效地、一次性地为一批长度生成指数衰减权重。
    此版本完全向量化，避免了Python循环。

    Args:
        lens (list[int]): 包含多个长度的列表。
        start_val (float): 第0个位置的权重值。
        end_val (float): 权重衰减趋近的最终值。
        decay_reach_percent (float): 权重衰减到接近最终值的点，占总长度的百分比。
        padding_value (float): 用于填充无效位置的数值。
        device: The desired device of the output tensor.

    Returns:
        torch.Tensor: 一个形状为 (len(lens), max(lens)) 的2D权重张量。
    """
    if not lens:
        return torch.empty(0, 0, device=device)

    # 1. 准备工作: 获取批次大小、最大长度，并将lens转为张量
    batch_size = len(lens)
    max_len = max(lens)
    lens_tensor = torch.tensor(lens, dtype=torch.float32, device=device)

    # 2. 向量化计算每个序列的衰减率 `decay_rate`
    # 注意：这里的每个变量都是一个向量，长度为 batch_size
    amplitude = start_val - end_val
    # 减1是为了得到正确的索引范围 [0, length-1]
    # 使用 clamp(min=1) 避免 length 为 1 时出现除以 0 的情况
    decay_end_index = (lens_tensor - 1).clamp(min=1) * decay_reach_percent
    
    # decay_rate 是一个形状为 (batch_size,) 的张量
    decay_rate = -torch.log(torch.tensor(0.01, device=device)) / decay_end_index
    
    # 3. 创建二维的 indices 和 decay_rate 以利用广播机制
    # indices 的形状: (max_len,) -> [0, 1, ..., max_len-1]
    indices = torch.arange(max_len, device=device)
    
    # decay_rate 的形状: (batch_size,) -> (batch_size, 1)
    # 这样它就可以和 indices [max_len,] 进行广播，结果形状为 (batch_size, max_len)
    exponent = -decay_rate.unsqueeze(1) * indices
    
    # 4. 一次性计算所有权重（这是一个完整的 BxL 矩阵）
    calculated_weights = amplitude * torch.exp(exponent) + end_val
    
    # 5. 创建掩码 (mask) 以将无效位置的值设为 padding_value
    # mask 的形状: (batch_size, max_len)
    mask = indices < lens_tensor.unsqueeze(1)
    
    # 6. 应用掩码，只保留有效长度内的权重值
    final_weights = torch.where(mask, calculated_weights, torch.tensor(padding_value, device=device))
    
    return final_weights