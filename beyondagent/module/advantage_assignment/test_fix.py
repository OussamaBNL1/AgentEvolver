from typing import List, Sequence, Optional
import torch
from copy import deepcopy

class PRMHyper:
    """PRM超参数配置"""
    def __init__(self):
        self.good_step_reward = 0.5
        self.bad_step_reward = -0.5

def grpo_advantage_process_steps_no_center(
    step_rewards_list: Sequence[Sequence[float]],
    eps: float = 1e-8,
) -> List[List[float]]:
    """
    GRPO优势处理函数（标准版本，会减均值）
    
    输入：
      step_rewards_list: 长度为 G 的序列；第 i 个元素是一段长度为 S_i 的列表/张量，
        只包含该样本每个"推理步骤"的 process reward（不含任何索引）。
     
    处理：
      1) 组内标准化：把所有样本、所有步骤的奖励拼在一起做 (r - mean)/std。
      2) 按步骤求"后缀和"：第 k 步的 advantage = 从第 k 步到最后所有步骤的标准化奖励之和。
     
    输出：
      与输入等长的列表；第 i 个元素是长度为 S_i 的列表，
      表示该样本每一步的 advantage。
    """
    if len(step_rewards_list) == 0:
        return []
    
    # to tensors
    step_rewards_tensors = [torch.as_tensor(r, dtype=torch.float32) for r in step_rewards_list]
    
    # group normalization (标准化：减均值除标准差)
    all_r = torch.cat([t.reshape(-1) for t in step_rewards_tensors], dim=0)
    mean = all_r.mean()
    std = all_r.std(unbiased=False)
    
    if std <= eps:
        norm_list = [torch.zeros_like(t) for t in step_rewards_tensors]
    else:
        norm_list = [(t - mean) / (std + eps) for t in step_rewards_tensors]
    
    # per-sample step-level advantages: suffix sums
    advantages_step: List[List[float]] = []
    for nr in norm_list:
        # suffix sum = reverse -> cumsum -> reverse
        adv = torch.flip(torch.cumsum(torch.flip(nr, dims=[0]), dim=0), dims=[0]).tolist()
        advantages_step.append(adv)
    
    return advantages_step

def compute_step_rewards_from_flags_consistent(
    orm_scores: Sequence[float],
    step_flags_list: Sequence[Sequence[bool]], 
    hyper: PRMHyper
) -> List[List[float]]:
    """
    一致性瓜分方式构造PRM奖励
    
    输入：
      orm_scores: 每个样本的ORM分数
      step_flags_list: 每个样本每个步骤的True/False标记
      hyper: 超参数配置
      
    输出：
      每个样本每个步骤的奖励值，ORM分数会加到最后一个步骤上
    """
    step_rewards_list = []
    
    for i, (orm_score, flags) in enumerate(zip(orm_scores, step_flags_list)):
        # 基础奖励：True步骤为正，False步骤为负
        rewards = []
        for flag in flags:
            if flag:  # True = good step
                rewards.append(hyper.good_step_reward)
            else:     # False = bad step  
                rewards.append(hyper.bad_step_reward)
        
        # 将ORM分数加到最后一个步骤上
        if len(rewards) > 0:
            rewards[-1] += orm_score
            
        step_rewards_list.append(rewards)
    
    return step_rewards_list

def check_order_per_sample(flags: List[bool], rewards: List[float], orm_score: float):
    """检查单个样本的奖励顺序是否符合预期"""
    print(f"  检查: Good步骤数={sum(flags)}, Bad步骤数={len(flags)-sum(flags)}, 最终ORM={orm_score}")
    
    # 检查前面步骤的基础奖励
    for i, (flag, reward) in enumerate(zip(flags[:-1], rewards[:-1])):
        expected = 0.5 if flag else -0.5
        if abs(reward - expected) > 1e-6:
            print(f"    警告: 步骤{i} flag={flag}, 期望={expected}, 实际={reward:.4f}")
    
    # 检查最后一步（包含ORM）
    if len(flags) > 0:
        last_flag = flags[-1]
        last_reward = rewards[-1]
        expected_base = 0.5 if last_flag else -0.5
        expected_final = expected_base + orm_score
        if abs(last_reward - expected_final) > 1e-6:
            print(f"    警告: 最后步骤 flag={last_flag}, 期望={expected_final}, 实际={last_reward:.4f}")

# 测试用例
def test_prm_grpo_consistent():
    print("=" * 60)
    print("一致性瓜分 PRM + 标准 GRPO 后缀和测试")
    print("=" * 60)
    
    hyper = PRMHyper()

    # 测试1: 成功轨迹批次，多数good步骤
    print("\n【测试1: 成功轨迹批次，多数GOOD步骤】")
    orms1 = [1.0, 1.0]
    flags1 = [
        [True, True, False, True, True, False, True, True, True],  # 7G,2B
        [True, True, True, False, True, True, False]               # 5G,2B
    ]
    step_rewards1 = compute_step_rewards_from_flags_consistent(orms1, flags1, hyper)
    advantages1 = grpo_advantage_process_steps_no_center(step_rewards1)
    for i in range(len(orms1)):
        print(f"样本{i} (ORM={orms1[i]}):")
        print(f"  Flags: {flags1[i]}")
        print(f"  Rewards: {[f'{x:.4f}' for x in step_rewards1[i]]}")
        print(f"  Advantages: {[f'{x:.4f}' for x in advantages1[i]]}")
        check_order_per_sample(flags1[i], step_rewards1[i], orms1[i])

    # 测试2: 失败轨迹批次，多数bad步骤
    print("\n【测试2: 失败轨迹批次，多数BAD步骤】")
    orms2 = [-1.0, -1.0]
    flags2 = [
        [True, False, False, True, False, False, False, False, False],  # 2G,7B
        [False, True, False, False, False, False]                       # 1G,5B
    ]
    step_rewards2 = compute_step_rewards_from_flags_consistent(orms2, flags2, hyper)
    advantages2 = grpo_advantage_process_steps_no_center(step_rewards2)
    for i in range(len(orms2)):
        print(f"样本{i} (ORM={orms2[i]}):")
        print(f"  Flags: {flags2[i]}")
        print(f"  Rewards: {[f'{x:.4f}' for x in step_rewards2[i]]}")
        print(f"  Advantages: {[f'{x:.4f}' for x in advantages2[i]]}")
        check_order_per_sample(flags2[i], step_rewards2[i], orms2[i])

    # 测试3: 混合批次
    print("\n【测试3: 混合批次】")
    orms3 = [1.0, -1.0]
    flags3 = [
        [True, True, False, True, True],      # 成功: 4G,1B
        [False, False, True, False, False]    # 失败: 1G,4B
    ]
    step_rewards3 = compute_step_rewards_from_flags_consistent(orms3, flags3, hyper)
    advantages3 = grpo_advantage_process_steps_no_center(step_rewards3)
    for i in range(len(orms3)):
        print(f"\n样本{i} (ORM={orms3[i]}):")
        print(f"  Flags: {flags3[i]}")
        print(f"  Rewards: {[f'{x:.4f}' for x in step_rewards3[i]]}")
        print(f"  Advantages: {[f'{x:.4f}' for x in advantages3[i]]}")
        check_order_per_sample(flags3[i], step_rewards3[i], orms3[i])

    # 测试4: 边界情况
    print("\n【测试4: 边界情况】")
    print("4a. 全GOOD：")
    orms4a = [1.0, 1.0]
    flags4a = [[True, True, True], [True, True, True, True]]
    step_rewards4a = compute_step_rewards_from_flags_consistent(orms4a, flags4a, hyper)
    advantages4a = grpo_advantage_process_steps_no_center(step_rewards4a)
    for i in range(len(orms4a)):
        print(f"  样本{i} (ORM={orms4a[i]}):")
        print(f"    Rewards: {[f'{x:.4f}' for x in step_rewards4a[i]]}")
        print(f"    Advantages: {[f'{x:.4f}' for x in advantages4a[i]]}")
        check_order_per_sample(flags4a[i], step_rewards4a[i], orms4a[i])

    print("4b. 全BAD：")
    orms4b = [-1.0, -1.0]
    flags4b = [[False, False, False], [False, False, False, False]]
    step_rewards4b = compute_step_rewards_from_flags_consistent(orms4b, flags4b, hyper)
    advantages4b = grpo_advantage_process_steps_no_center(step_rewards4b)
    for i in range(len(orms4b)):
        print(f"  样本{i} (ORM={orms4b[i]}):")
        print(f"    Rewards: {[f'{x:.4f}' for x in step_rewards4b[i]]}")
        print(f"    Advantages: {[f'{x:.4f}' for x in advantages4b[i]]}")
        check_order_per_sample(flags4b[i], step_rewards4b[i], orms4b[i])

if __name__ == "__main__":
    test_prm_grpo_consistent()