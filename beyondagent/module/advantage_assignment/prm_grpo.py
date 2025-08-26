# -*- coding: utf-8 -*-
# PRM (step-level) → group z-score (step-level) → per-trajectory reprojection → suffix-sum (step-level) → broadcast to token
from __future__ import annotations
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import torch

@dataclass
class PRMHyper:
    # 权重：一致性步的权重大，不一致性步的权重小
    consistent_scale: float = 1.0
    pos_unconsistent_scale: float = 0.2   # 成功轨迹里的 BAD 步权重
    neg_unconsistent_scale: float = 0.2   # 失败轨迹里的 GOOD 步权重
    eps: float = 1e-8
    do_batch_zscore: bool = True          # 是否做组内 z-score（按 step 级）

# ----------------------------- Utils -----------------------------

def _ensure_tensor(x, device, dtype=None):
    if torch.is_tensor(x):
        t = x.to(device=device)
        if dtype is not None:
            t = t.to(dtype)
        return t
    return torch.as_tensor(x, device=device, dtype=dtype)

def _num_steps_from_step_ids(step_ids_row: torch.Tensor) -> int:
    """step_ids: shape (L,) with -1 for non-response tokens; contiguous step ids starting at 0."""
    if step_ids_row.numel() == 0:
        return 0
    m = torch.amax(step_ids_row)
    return int(m.item() + 1) if m.item() >= 0 else 0

def _align_flags(flags: List[bool], K: int, is_success: bool) -> List[bool]:
    if len(flags) == K:
        return list(flags)
    default_flag = True if is_success else False
    if len(flags) < K:
        return list(flags) + [default_flag] * (K - len(flags))
    else:
        return list(flags[:K])

# ------------------------- Core (Plan-3) -------------------------

def compute_step_rewards_from_flags_consistent_centered(
    orms_sign: torch.Tensor,
    step_flags: List[List[bool]],
    step_ids: torch.Tensor,
    group_ids: torch.Tensor,
    hyper: PRMHyper,
) -> List[List[float]]:
    """
    方案3（按 step）：
      1) 一致性权重瓜分：r_raw，逐轨迹 ∑=ORM_sign (±1)
      2) 组内（group）step-level z-score：r_std
      3) 逐轨迹去均值 + 加回 ORM_sign/K：r_proj（逐轨迹 ∑=ORM_sign）
    返回：r_proj（逐轨迹按 step 的回报）
    """
    device = step_ids.device
    B, L = step_ids.shape
    assert orms_sign.shape[0] == B
    assert group_ids.shape[0] == B

    # ---- 1) 一致性瓜分（逐轨迹 ∑=ORM_sign）----
    step_rewards_raw: List[List[float]] = []
    Ks: List[int] = []
    for i in range(B):
        K = _num_steps_from_step_ids(step_ids[i])
        Ks.append(K)
        if K == 0:
            step_rewards_raw.append([])
            continue

        is_success = bool(orms_sign[i].item() > 0)
        flags_i = _align_flags(step_flags[i] if i < len(step_flags) else [], K, is_success=is_success)

        if is_success:
            w_good = hyper.consistent_scale
            w_bad  = hyper.pos_unconsistent_scale
        else:
            w_good = hyper.neg_unconsistent_scale
            w_bad  = hyper.consistent_scale

        weights = torch.tensor([w_good if f else w_bad for f in flags_i], device=device, dtype=torch.float32)
        total_w = float(weights.sum().item())
        if total_w <= hyper.eps:
            # 退化保护：均分
            weights[:] = 1.0
            total_w = float(K)
        unit = float(orms_sign[i].item()) / total_w
        r_raw = (weights * unit).tolist()  # ∑=±1
        step_rewards_raw.append(r_raw)

    if not hyper.do_batch_zscore:
        return step_rewards_raw

    # ---- 2) 组内 step-level z-score ----
    # 先把每个 group 里所有 step 的 r_raw 拉平，求 mu, std，并对该组所有样本的 step 做 (x-mu)/std
    step_rewards_std: List[List[float]] = [[] for _ in range(B)]
    # 建立 group -> indices
    unique_groups = torch.unique(group_ids)
    for g in unique_groups.tolist():
        idxs = (group_ids == g).nonzero(as_tuple=False).view(-1).tolist()
        # flatten this group's steps
        flat_vals = []
        for i in idxs:
            flat_vals.extend(step_rewards_raw[i])
        if len(flat_vals) == 0:
            # 该组无step（极端情况）
            for i in idxs:
                step_rewards_std[i] = list(step_rewards_raw[i])
            continue
        t = torch.tensor(flat_vals, device=device, dtype=torch.float32)
        mu = float(t.mean().item())
        std = float(t.std(unbiased=False).item())
        if std <= hyper.eps:
            # 仅减均值
            for i in idxs:
                step_rewards_std[i] = [float(x - mu) for x in step_rewards_raw[i]]
        else:
            inv = 1.0 / (std + 1e-12)
            for i in idxs:
                step_rewards_std[i] = [float((x - mu) * inv) for x in step_rewards_raw[i]]

    # ---- 3) 逐轨迹再投影（∑=ORM_sign）----
    step_rewards_proj: List[List[float]] = []
    for i in range(B):
        K = Ks[i]
        if K == 0:
            step_rewards_proj.append([])
            continue
        ri = step_rewards_std[i]
        m = sum(ri) / K
        # 去均值 + 均分 ORM_sign/K
        base = float(orms_sign[i].item()) / K
        r_proj = [float(x - m + base) for x in ri]
        step_rewards_proj.append(r_proj)

    return step_rewards_proj

def suffix_sum_on_steps(step_rewards: List[List[float]]) -> List[List[float]]:
    """对每个样本的 step 回报做后缀和，输出同形状的 step-adv。"""
    adv: List[List[float]] = []
    for r in step_rewards:
        if not r:
            adv.append([])
            continue
        t = torch.tensor(r, dtype=torch.float32)
        s = torch.flip(torch.cumsum(torch.flip(t, dims=[0]), dim=0), dims=[0])
        adv.append([float(x) for x in s])
    return adv

def broadcast_step_adv_to_tokens(
    step_adv: List[List[float]],
    step_ids: torch.Tensor,
) -> torch.Tensor:
    """把 step-adv 按 step_ids 广播到 token 上。step_ids 为 -1 的位置填 0。"""
    device = step_ids.device
    B, L = step_ids.shape
    out = torch.zeros((B, L), device=device, dtype=torch.float32)
    for i in range(B):
        if not step_adv[i]:
            continue
        adv_i = torch.tensor(step_adv[i], device=device, dtype=torch.float32)
        # mask for response tokens
        sid_row = step_ids[i]
        valid = sid_row >= 0
        if torch.any(valid):
            sids = sid_row[valid]
            out[i, valid] = adv_i[sids]
    return out

# ----------------------------- Entry -----------------------------

def compute_prm_grpo_advantages(
    batch,                          # DataProto 或兼容结构：batch.batch[...] 可索引
    step_flags: List[List[bool]],   # 每条轨迹的 GOOD/BAD 标志，长度与 step 数匹配（不足则按 ORM 符号补齐）
    hyper: Optional[PRMHyper] = None,
) -> Dict[str, torch.Tensor]:
    """
    方案3 + ORM 强制为 ±1 的版本：
      - ORM_sign = sign(sum(token_level_rewards))
      - 在 step 上瓜分、标准化、再投影，得到 step-adv
      - 将 step-adv 按 step_ids 广播到 token 得到 (B, L) 的 advantages
    返回：
      - advantages: (B, L) token-level advantages
      - orm_scalar: (B,) 逐条轨迹的 ±1
    """
    if hyper is None:
        hyper = PRMHyper()

    # ---- 取必要字段 ----
    device = None
    # responses 仅用于确定设备/长度
    responses = batch.batch["responses"]
    if torch.is_tensor(responses):
        device = responses.device
    else:
        responses = torch.as_tensor(responses)
        device = responses.device

    step_ids = _ensure_tensor(batch.batch["step_ids"], device=device, dtype=torch.long)  # (B, L_resp) with -1 for non-response
    group_ids = _ensure_tensor(batch.batch["group_ids"], device=device, dtype=torch.long).view(-1)

    # 取 token-level reward（可能字段名不同，做兜底）
    token_keys_try = ["token_level_rewards", "response_token_level_rewards", "token_rewards"]
    token_level_rewards = None
    for k in token_keys_try:
        if k in batch.batch:
            token_level_rewards = _ensure_tensor(batch.batch[k], device=device, dtype=torch.float32)
            break
    if token_level_rewards is None:
        raise KeyError("token-level rewards not found in batch (tried keys: token_level_rewards / response_token_level_rewards / token_rewards)")

    # ---- ORM_sign = ±1 ----
    orm_sum = token_level_rewards.sum(dim=1)   # (B,)
    orms_sign = torch.where(orm_sum > 0, torch.ones_like(orm_sum), -torch.ones_like(orm_sum)).to(dtype=torch.float32)

    # ---- Step-level pipeline ----
    step_rewards_proj = compute_step_rewards_from_flags_consistent_centered(
        orms_sign=orms_sign,
        step_flags=step_flags,
        step_ids=step_ids,
        group_ids=group_ids,
        hyper=hyper,
    )
    step_adv = suffix_sum_on_steps(step_rewards_proj)
    advantages = broadcast_step_adv_to_tokens(step_adv, step_ids)

    return {
        "advantages": advantages,        # (B, L_resp)
        "orm_scalar": orms_sign,         # (B,)
    }
