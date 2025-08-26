from typing import List, Sequence
import torch
from dataclasses import dataclass

@dataclass
class PRMHyper:
    consistent_scale: float = 1.0
    pos_unconsistent_scale: float = 0.2  # 成功轨迹的 BAD 步骤权重
    neg_unconsistent_scale: float = 0.2  # 失败轨迹的 GOOD 步骤权重
    eps: float = 1e-8
    scale_by_std: bool = True            # 仅“正数缩放”，不做减均值

def compute_step_rewards_from_flags_consistent(
    orms: List[float],
    step_flags_list: List[List[bool]], 
    hyper: PRMHyper,
) -> List[List[float]]:
    """
    一致性瓜分式 PRM：
    - 严格保证每个轨迹 sum(step_rewards) == ORM
    - 成功轨迹：GOOD 步权重=1，BAD 步权重=α（<1）
      失败轨迹：BAD 步权重=1，GOOD 步权重=α（<1）
    - 不做减均值标准化，保持符号一致性
    """
    step_rewards_list: List[List[float]] = []
    
    for i, (orm, flags) in enumerate(zip(orms, step_flags_list)):
        S = len(flags)
        if S == 0:
            step_rewards_list.append([])
            continue

        # 轨迹成功/失败
        is_success = (orm >= 0.0)

        # 权重规则（consistent vs inconsistent）
        if is_success:
            w_good = hyper.consistent_scale                    # 1.0
            w_bad  = hyper.pos_unconsistent_scale             # 0.2
        else:
            w_good = hyper.neg_unconsistent_scale             # 0.2
            w_bad  = hyper.consistent_scale                   # 1.0

        # 权重列表
        weights = [(w_good if f else w_bad) for f in flags]
        total_w = sum(weights)
        unit = 0.0 if abs(total_w) <= hyper.eps else (orm / total_w)

        # 分配奖励（注意：orm 的符号直接体现在 unit 上）
        step_rewards = [w * unit for w in weights]
        step_rewards_list.append(step_rewards)

        # 验证总和
        actual_sum = sum(step_rewards)
        print(f"[PRM] 样本{i}: ORM={orm:.3f}, ∑step={actual_sum:.6f}, 差异={abs(orm-actual_sum):.8f}")

    # （可选）仅正数缩放：除以 batch 内的标准差，保持符号不变
    if hyper.scale_by_std:
        flat = [x for lst in step_rewards_list for x in lst]
        if len(flat) > 0:
            import math
            mean = sum(flat) / len(flat)   # 仅用于估算方差，不会用于减均值
            var = sum((x - mean) * (x - mean) for x in flat) / max(1, len(flat))
            std = math.sqrt(var + hyper.eps)
            if std > 0:
                step_rewards_list = [[x / std for x in lst] for lst in step_rewards_list]

    return step_rewards_list

def grpo_advantage_process_steps_no_center(
    step_rewards_list: Sequence[Sequence[float]],
) -> List[List[float]]:
    """
    GRPO 优势：仅后缀和（不做减均值标准化，确保失败轨迹 advantage 保持为负）
    """
    advantages_list: List[List[float]] = []
    for rewards in step_rewards_list:
        t = torch.as_tensor(rewards, dtype=torch.float32)
        adv = torch.flip(torch.cumsum(torch.flip(t, dims=[0]), dim=0), dims=[0]).tolist()
        advantages_list.append(adv)
    return advantages_list

# =============== 小工具：一致性与序关系检查 ===============

def _avg(xs: List[float]) -> float:
    return sum(xs) / max(1, len(xs))

def check_order_per_sample(flags: List[bool], rewards: List[float], orm: float):
    good_vals = [r for r, f in zip(rewards, flags) if f]
    bad_vals  = [r for r, f in zip(rewards, flags) if not f]
    if len(good_vals) == 0 or len(bad_vals) == 0:
        return  # 无法比较时跳过
    ag, ab = _avg(good_vals), _avg(bad_vals)
    # 统一用“数值大小”比较（同号下，good 应该更大）
    ok = ag >= ab
    sign_ok = all((r >= 0) for r in rewards) if orm >= 0 else all((r <= 0) for r in rewards)
    print(f"    [CHECK] avg(GOOD)={ag:.4f} >= avg(BAD)={ab:.4f} ? {'OK' if ok else 'FAIL'}; "
          f"sign={'OK' if sign_ok else 'FAIL'}")

# =========================== 测试用例 ===========================

def test_prm_grpo_consistent():
    print("=" * 60)
    print("一致性瓜分 PRM + 无减均值 GRPO 后缀和（保号）测试")
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
