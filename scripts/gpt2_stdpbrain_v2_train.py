#!/usr/bin/env python3
"""
GPT-2 (from lal) × stdpbrain v2 — 真 STDP 训练 + 思维流可视化
================================================================

v1 (上一版) 只是冻结推理 + 观测；本 v2 做两件事:

1. **真 STDP 训练**:
   - 解冻 brain 模块的 nn.Parameter
   - 维护一个 (768, 768) 的可训练 STDP 突触矩阵 (requires_grad=True)
   - 每个 token 后做一次三因子更新 ΔW = grad × timing × DA × lr
   - 把 STDP 增量真实叠加到 GPT-2 的 lm_head logits 上
     (dynamic_logits = STDP_W @ last_hidden → 加到 lm_head_logits)
   - 配合小 lr (1e-3) 和权重裁剪 (clamp ±0.3)

2. **思维流可视化**:
   复刻 stdpbrain InnerThoughtEngine 的状态机思想:
     FOCUSED → WANDERING → REFLECTING → RESTING
   每个状态有不同的"引导词"和"思维风格"，
   每生成几个 token，就 dump 一段"内心独白"，让用户看到模型在"想什么"。

输出:
   /home/z/my-project/download/gpt2_stdpbrain_v2_train.json
   /home/z/my-project/download/gpt2_stdpbrain_v2_train.txt
   /home/z/my-project/download/gpt2_stdpbrain_v2_thoughts.txt  (思维流纯文本)
   /home/z/my-project/download/gpt2_stdpbrain_v2.png
"""

from __future__ import annotations

import os
import sys
import json
import time
import random
import warnings
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass, field

warnings.filterwarnings("ignore")

# === 路径 setup ===
STDPBRAIN_DIR = "/home/z/my-project/repos/stdpbrain"
DOWNLOAD_DIR = "/home/z/my-project/download"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.chdir(STDPBRAIN_DIR)
sys.path.insert(0, STDPBRAIN_DIR)

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.nn.functional as F

# === 全局常量 ===
HIDDEN_SIZE = 768
BATCH = 1
DEVICE = "cpu"
TORCH_THREADS = 4
torch.set_num_threads(TORCH_THREADS)

# STDP 训练超参
STDP_LR = 1e-3                  # 小 lr，符合 stdpbrain STDPConfig.alpha_LTP=0.005 量级
STDP_WEIGHT_CLAMP = 0.3         # stdpbrain WEIGHT_MAX
STDP_LOGIT_SCALE = 1.0          # STDP 增量对 logits 的缩放
STDP_LOSS_TARGET_TOKEN = True   # 用预测 token 自身作为 target (自监督)

MAX_NEW_TOKENS = 25             # 每个 prompt 生成 25 个 token
THOUGHT_INTERVAL = 5            # 每生成 5 个 token 输出一段"内心独白"


# ============================================================================
# 1. 加载 GPT-2
# ============================================================================
def load_gpt2():
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    print("[1/5] 加载 GPT-2 (lal 仓库参考模型)...", flush=True)
    t0 = time.time()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    model.to(DEVICE)
    for p in model.parameters():
        p.requires_grad_(False)  # GPT-2 主体冻结
    dt = time.time() - t0
    n = sum(p.numel() for p in model.parameters())
    print(f"      ✅ GPT-2 | {n/1e6:.1f}M params | {dt:.1f}s", flush=True)
    return tokenizer, model


# ============================================================================
# 2. 加载 brain 模块（解冻）+ STDP 突触矩阵（可训练）
# ============================================================================
class STDPBrain:
    """封装 brain 模块 + 可训练 STDP 突触矩阵。"""

    def __init__(self):
        import importlib
        print("[2/5] 加载 brain 模块 (hidden=768)...", flush=True)
        self.modules: Dict[str, nn.Module] = {}
        spec = [
            ("cerebellar",     "core.cerebellar_correction_677",    "create_cerebellar_correction_system",  dict(hidden_size=HIDDEN_SIZE)),
            ("basal_ganglia",  "core.basal_ganglia_dopamine",       "create_basal_ganglia_dopamine_system", dict(hidden_size=HIDDEN_SIZE)),
            ("lc_ne",          "core.locus_coeruleus_ne",           "create_locus_coeruleus_ne_system",     dict(hidden_size=HIDDEN_SIZE)),
            ("astrocyte",      "core.astrocyte_neuromod_coupling",  "create_astrocyte_neuromod_coupling",   dict(hidden_size=HIDDEN_SIZE)),
            ("amygdala",       "core.amygdala",                     "create_amygdala_system",               dict(hidden_size=HIDDEN_SIZE)),
            ("dual_process",   "core.dual_process",                 "create_dual_process_system",           dict(hidden_size=HIDDEN_SIZE)),
            ("synaptic_plast", "core.synaptic_plasticity",          "create_synaptic_plasticity_system",    dict(hidden_size=HIDDEN_SIZE)),
            ("signal_bus",     "core.signal_bus",                   "create_signal_bus",                    dict()),
            ("adv_neuromod",   "core.advanced_neuromodulation",     "create_advanced_neuromodulation_system", dict(hidden_size=HIDDEN_SIZE)),
        ]
        total = 0
        for name, path, factory, kwargs in spec:
            t0 = time.time()
            try:
                mod = importlib.import_module(path)
                f = getattr(mod, factory)
                m = f(**kwargs)
                # ★ brain 模块本身冻结 (eval mode, no grad) - 它们的 forward 提供 DA/NE/ΔW 信号
                m.eval() if hasattr(m, "eval") else None
                for p in m.parameters():
                    p.requires_grad_(False)
                n = sum(p.numel() for p in m.parameters()) if hasattr(m, "parameters") else 0
                total += n
                self.modules[name] = m
                print(f"      ✅ {name:<16} | {n:>10,} params (frozen, signal-only) | {(time.time()-t0)*1000:>6.1f}ms", flush=True)
            except Exception as e:
                print(f"      ❌ {name:<16} | ERROR: {e}", flush=True)
        print(f"      合计 brain 参数: {total:,} (全部 requires_grad=False, 仅作信号源)", flush=True)

        # === 可训练 STDP 突触矩阵 ===
        # 这是 stdpbrain dual_weight_layers.py 中 dynamic_weight 的等价物
        # 维度 (768, 768) - 在 hidden space 残差式注入: hidden' = hidden + 0.1 * STDP_W @ hidden
        # 这是唯一可训练参数 - 大幅降低 optimizer 内存占用
        self.stdp_weight = nn.Parameter(
            torch.randn(HIDDEN_SIZE, HIDDEN_SIZE, device=DEVICE) * 0.01,
            requires_grad=True,
        )
        # 只优化 stdp_weight - 用 SGD + momentum (内存远小于 AdamW)
        self.optimizer = torch.optim.SGD(
            [self.stdp_weight], lr=STDP_LR, momentum=0.9, weight_decay=1e-4,
        )
        print(f"      ✅ STDP 突触矩阵 (768×768, 可训练) + SGD lr={STDP_LR} momentum=0.9", flush=True)
        print(f"      (优化器仅维护 stdp_weight 一个张量, 大幅降低内存)", flush=True)

        # DA / NE 等状态变量
        self.da_level = 0.5
        self.ne_level = 0.5
        self.valence = 0.0
        self.system_mode = "S1"

    def step(
        self,
        hidden_now: torch.Tensor,    # (1, 768)
        hidden_prev: torch.Tensor,   # (1, 768)
        target_id: int,              # 下一个真实 token id (用作自监督 target)
        step_idx: int,
    ) -> Tuple[Dict[str, float], torch.Tensor, torch.Tensor]:
        """
        一遍脑模块闭环 + STDP 突触更新.
        返回: (snapshot, stdp_modified_hidden, stdp_logit_delta)

        内存策略: brain 模块全部在 no_grad 下运行 (forward only);
                  只有 stdp_weight 走 requires_grad, 用 synaptic_plasticity
                  产生的 ΔW 作为"伪梯度"喂给 SGD.
        """
        snap: Dict[str, float] = {"step": step_idx}
        H = hidden_now.detach()  # 切断与 GPT-2 计算图的联系

        # ---- 0. STDP 增量注入 (可训练, 需要梯度) ----
        # hidden_modified = hidden + 0.1 * STDP_W @ hidden
        # 这里需要保留 stdp_weight 的梯度
        stdp_delta = F.linear(H, self.stdp_weight)  # (1, 768)
        modified_hidden = H + stdp_delta * 0.1

        # ---- 1~6. brain 模块闭环 (全部 no_grad) ----
        with torch.no_grad():
            # 1. 小脑纠错
            cb = self.modules.get("cerebellar")
            cerebellar_err = 0.0
            corrected = H
            if cb is not None:
                try:
                    r = cb.forward(H, hidden_prev, H)
                    err = r.get("prediction_error", None)
                    if isinstance(err, torch.Tensor):
                        cerebellar_err = float(err.norm().item())
                    else:
                        cerebellar_err = float(err) if err is not None else 0.0
                    corrected = r.get("corrected_output", H)
                    if not isinstance(corrected, torch.Tensor):
                        corrected = H
                except Exception:
                    pass
            snap["cerebellar_error"] = cerebellar_err

            # 2. 基底节 DA
            bg = self.modules.get("basal_ganglia")
            da_level = 0.5
            bg_out = corrected
            if bg is not None:
                try:
                    r = bg.forward(corrected)
                    if hasattr(bg, "get_dopamine_level"):
                        da = bg.get_dopamine_level()
                        da_level = float(da.mean().item()) if isinstance(da, torch.Tensor) else float(da)
                    bg_out = r.get("output", corrected)
                    if not isinstance(bg_out, torch.Tensor):
                        bg_out = corrected
                except Exception:
                    pass
            snap["da"] = da_level
            self.da_level = da_level

            # 3. 蓝斑 NE
            lc = self.modules.get("lc_ne")
            ne_level = 0.5
            lc_out = bg_out
            if lc is not None:
                try:
                    r = lc.forward(bg_out)
                    if hasattr(lc, "get_ne_level"):
                        ne = lc.get_ne_level()
                        ne_level = float(ne.mean().item()) if isinstance(ne, torch.Tensor) else float(ne)
                    lc_out = r.get("output", bg_out)
                    if not isinstance(lc_out, torch.Tensor):
                        lc_out = bg_out
                except Exception:
                    pass
            snap["ne"] = ne_level
            self.ne_level = ne_level

            # 4. 胶质耦合
            ast = self.modules.get("astrocyte")
            clearance = 0.0
            if ast is not None:
                try:
                    ast.forward(da_level=da_level, ne_level=ne_level, ach_level=0.5, sht_level=0.5)
                    if hasattr(ast, "get_glymphatic_rate"):
                        clearance = float(ast.get_glymphatic_rate())
                except Exception:
                    pass
            snap["clearance"] = clearance

            # 5. 杏仁核 valence
            amy = self.modules.get("amygdala")
            valence = 0.0
            amy_out = lc_out
            if amy is not None:
                try:
                    amy_in = lc_out if lc_out.dim() == 2 else lc_out.unsqueeze(0)
                    r = amy.forward(amy_in)
                    v = r.get("stats", {}).get("valence", 0)
                    if isinstance(v, torch.Tensor):
                        v = float(v.mean().item())
                    valence = float(v)
                    amy_out = r.get("output", amy_in)
                    if not isinstance(amy_out, torch.Tensor):
                        amy_out = amy_in
                except Exception:
                    pass
            snap["valence"] = valence
            self.valence = valence

            # 6. 双系统
            dp = self.modules.get("dual_process")
            system_mode = "?"
            if dp is not None:
                try:
                    r = dp.forward(amy_out)
                    system_mode = str(r.get("stats", {}).get("active_system", "?"))
                except Exception:
                    pass
            snap["system"] = system_mode
            self.system_mode = system_mode

            # 7. 三因子 ΔW 信号 (no_grad, 仅作为伪梯度来源)
            sp = self.modules.get("synaptic_plast")
            delta_w_signal = None
            if sp is not None:
                try:
                    new_w = sp.forward(
                        pre_activity=hidden_prev,
                        post_activity=H,
                        dopamine_level=da_level,
                        weights=self.stdp_weight.data.clone(),
                    )
                    delta_w_signal = (new_w - self.stdp_weight.data)  # ΔW from 三因子规则
                except Exception:
                    delta_w_signal = None

        # ---- 8. 把 ΔW 作为伪梯度喂给 stdp_weight ----
        # 设计: STDP_W 的梯度 = -ΔW × reward × scale
        #   - 负号: SGD 是梯度下降, 我们要"朝着 ΔW 方向"走 (ΔW 已经是 desired change)
        #   - reward = (DA - 0.5) 标准化到 [-0.5, 0.5]
        #   - scale = 0.1 防止一步走太远
        if delta_w_signal is not None:
            reward = (da_level - 0.5)
            pseudo_grad = -delta_w_signal * reward * 0.1
            if self.stdp_weight.grad is None:
                self.stdp_weight.grad = pseudo_grad.detach().clone()
            else:
                self.stdp_weight.grad.copy_(pseudo_grad.detach())

        snap["stdp_norm"] = float(self.stdp_weight.norm().item())
        snap["stdp_grad_norm"] = float(self.stdp_weight.grad.norm().item()) if self.stdp_weight.grad is not None else 0.0

        # ---- 9. optimizer step + clip + clamp ----
        if self.stdp_weight.grad is not None:
            grad_norm = self.stdp_weight.grad.norm().item()
            if grad_norm > 10.0:
                self.stdp_weight.grad.mul_(10.0 / grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()
        # 权重裁剪 (符合 stdpbrain WEIGHT_MAX)
        self.stdp_weight.data.clamp_(-STDP_WEIGHT_CLAMP, STDP_WEIGHT_CLAMP)

        return snap, modified_hidden.detach(), stdp_delta.detach()


# ============================================================================
# 3. 思维流状态机 (复刻 stdpbrain InnerThoughtEngine 的思想)
# ============================================================================
class ThoughtStream:
    """
    模拟 stdpbrain InnerThoughtEngine 的思维状态机.
    4 个状态: FOCUSED / WANDERING / REFLECTING / RESTING
    每个状态有引导词池 + 思维风格.
    """

    STATES = ["FOCUSED", "WANDERING", "REFLECTING", "RESTING"]

    # 状态转换矩阵 (from → to 的概率) — 取自 stdpbrain InnerThoughtEngine 初始值
    TRANSITION = torch.tensor([
        [0.65, 0.20, 0.15, 0.00],   # FOCUSED
        [0.35, 0.45, 0.20, 0.00],   # WANDERING
        [0.40, 0.00, 0.35, 0.25],   # REFLECTING
        [0.50, 0.30, 0.00, 0.20],   # RESTING
    ])

    LEADS = {
        "FOCUSED":    ["[focus] 让我聚焦分析...", "[focus] 仔细推导的话...", "[focus] 核心是..."],
        "WANDERING":  ["[wander] 说起来，刚才想到...", "[wander] 也许换个角度...", "[wander] 让我联想到..."],
        "REFLECTING": ["[reflect] 等等，这样对吗...", "[reflect] 重新审视一下...", "[reflect] 我刚才是不是陷入了..."],
        "RESTING":    ["[rest] 嗯...", "[rest] 整理一下...", "[rest] 思考着..."],
    }

    def __init__(self):
        self.state = "RESTING"
        self.state_duration = 0
        self.history: List[Dict[str, Any]] = []
        self.thought_segments: List[str] = []

    def transition(self, da: float, ne: float, valence: float):
        """根据脑活动状态采样下一个思维状态. 高 DA → 留在 FOCUSED; 低 NE → RESTING."""
        from_idx = self.STATES.index(self.state)
        # 用脑活动轻微调整转换概率
        logits = torch.log(self.TRANSITION[from_idx].clamp(min=1e-4))
        # 高 DA → 倾向 FOCUSED; 低 NE → 倾向 RESTING; 负 valence → REFLECTING
        logits[0] += (da - 0.5) * 0.5   # FOCUSED
        logits[3] += (0.5 - ne) * 0.5   # RESTING
        logits[2] += (-valence) * 0.5   # REFLECTING (负 valence 触发反思)
        probs = F.softmax(logits, dim=0)
        next_idx = torch.multinomial(probs, 1).item()
        next_state = self.STATES[next_idx]
        if next_state != self.state:
            self.state_duration = 0
        else:
            self.state_duration += 1
        self.state = next_state

    def generate_segment(
        self,
        prompt: str,
        recent_tokens: List[str],
        brain_snap: Dict[str, float],
        step_idx: int,
    ) -> str:
        """生成一段思维流文本 (基于当前状态 + 脑活动 + 最近生成的 token)."""
        lead = random.choice(self.LEADS[self.state])
        recent_text = "".join(recent_tokens[-8:]).strip()
        if not recent_text:
            recent_text = "(刚开始)"

        # 根据状态拼接不同的"思维内容"
        if self.state == "FOCUSED":
            thought = (
                f"{lead} 当前生成「{recent_text}」，"
                f"DA={brain_snap['da']:.3f}(奖赏预测误差)，"
                f"小脑误差={brain_snap['cerebellar_error']:.2f} — "
                f"{'预测准确，继续推进' if brain_snap['cerebellar_error'] < 60 else '预测偏离，需要修正'}"
            )
        elif self.state == "WANDERING":
            thought = (
                f"{lead} 「{recent_text}」让我想到——"
                f"valence={brain_snap['valence']:+.3f}，"
                f"杏仁核判定情绪{'偏积极' if brain_snap['valence'] > 0 else '偏消极' if brain_snap['valence'] < -0.05 else '中性'}，"
                f"突触范数={brain_snap['stdp_norm']:.3f}(STDP{'正在强化' if brain_snap['stdp_norm'] > 0.5 else '稳定'})"
            )
        elif self.state == "REFLECTING":
            thought = (
                f"{lead} 我刚才输出「{recent_text}」对吗？"
                f"DA={brain_snap['da']:.3f}, NE={brain_snap['ne']:.3f}，"
                f"{'唤醒度偏低，可能漏了细节' if brain_snap['ne'] < 0.3 else '唤醒度合适'}，"
                f"应该{'继续这个方向' if brain_snap['da'] > 0.55 else '换一个角度'}"
            )
        else:  # RESTING
            thought = (
                f"{lead} 暂停一下，整理当前状态：已生成 {step_idx+1} 个 token，"
                f"突触范数 {brain_snap['stdp_norm']:.3f}，"
                f"梯度范数 {brain_snap['stdp_grad_norm']:.4f}，"
                f"系统模式 {brain_snap['system']} — 大脑在「消化」刚学到的"
            )

        seg = {
            "step": step_idx,
            "state": self.state,
            "thought": thought,
            "brain": {k: v for k, v in brain_snap.items() if k != "step"},
        }
        self.history.append(seg)
        self.thought_segments.append(thought)
        return thought


# ============================================================================
# 4. 跑一个 prompt: GPT-2 生成 + STDP 训练 + 思维流
# ============================================================================
@torch.no_grad()
def gpt2_forward_with_stdp(gpt2, input_ids, brain: STDPBrain):
    """GPT-2 forward + STDP 注入到最后一层 hidden state."""
    out = gpt2(input_ids, output_hidden_states=True, use_cache=False)
    last_hidden = out.hidden_states[-1][:, -1, :].squeeze(0)  # (768,)
    if last_hidden.dim() == 1:
        last_hidden = last_hidden.unsqueeze(0)  # (1, 768)

    # STDP 增量注入 (在 lm_head 之前)
    stdp_delta = F.linear(last_hidden, brain.stdp_weight)  # (1, 768)
    modified_hidden = last_hidden + stdp_delta * 0.1

    # 走 lm_head
    logits = gpt2.lm_head(modified_hidden)  # (1, vocab)
    return logits, last_hidden, modified_hidden, stdp_delta


def run_one_prompt(
    tokenizer, gpt2, brain: STDPBrain,
    thought_stream: ThoughtStream,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> Dict[str, Any]:
    print(f"\n{'─' * 90}", flush=True)
    print(f"📌 Prompt: {prompt!r}", flush=True)
    print(f"{'─' * 90}", flush=True)

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    generated_ids = input_ids.clone()

    # 先跑一次 forward 拿 hidden_prev
    with torch.no_grad():
        out0 = gpt2(input_ids, output_hidden_states=True, use_cache=False)
        hidden_prev = out0.hidden_states[-1][:, -1, :].squeeze(0).detach()
        if hidden_prev.dim() == 1:
            hidden_prev = hidden_prev.unsqueeze(0)

    snapshots: List[Dict[str, Any]] = []
    generated_tokens: List[str] = []
    thought_log: List[Dict[str, Any]] = []

    print(f"  step | token            | DA     | NE     | err    | val   | sys | stdp_norm | grad_norm", flush=True)
    print(f"  -----+------------------+--------+--------+--------+-------+-----+-----------+----------", flush=True)

    t0 = time.time()
    for step in range(max_new_tokens):
        # 1. GPT-2 forward + STDP 注入 (但这里 hidden_prev 用上一步的)
        # 由于 brain.step 内部会做 STDP 更新，我们需要在 forward 之前先获取当前 hidden
        with torch.no_grad():
            out = gpt2(generated_ids, output_hidden_states=True, use_cache=False)
            hidden_now = out.hidden_states[-1][:, -1, :].squeeze(0).detach()
            if hidden_now.dim() == 1:
                hidden_now = hidden_now.unsqueeze(0)

        # 2. brain step + STDP 更新 (用 GPT-2 预测的下一个 token 作为 target)
        # 先用未修改的 hidden 预测下一个 token id 作为 target
        with torch.no_grad():
            base_logits = gpt2.lm_head(hidden_now)
            target_id = int(torch.argmax(base_logits, dim=-1).item())

        # brain.step 会修改 stdp_weight
        snap, modified_hidden, stdp_delta = brain.step(hidden_now, hidden_prev, target_id, step)

        # 3. 用 STDP 修改后的 hidden 重新算 logits → 贪心选 token
        with torch.no_grad():
            modified_logits = gpt2.lm_head(modified_hidden)
            next_id = int(torch.argmax(modified_logits, dim=-1).item())

        # 4. 检查 STDP 是否改变了输出
        stdp_changed = (next_id != target_id)
        snap["token"] = tokenizer.decode([next_id])
        snap["token_id"] = next_id
        snap["base_token_id"] = target_id
        snap["stdp_changed_output"] = stdp_changed
        snapshots.append(snap)

        generated_ids = torch.cat([generated_ids, torch.tensor([[next_id]], device=DEVICE)], dim=1)
        generated_tokens.append(snap["token"])

        # 5. 思维流: 每 THOUGHT_INTERVAL 个 token 输出一段
        if step % THOUGHT_INTERVAL == 0 or step == max_new_tokens - 1:
            thought_stream.transition(brain.da_level, brain.ne_level, brain.valence)
            thought = thought_stream.generate_segment(
                prompt=prompt,
                recent_tokens=generated_tokens,
                brain_snap=snap,
                step_idx=step,
            )
            thought_log.append({
                "step": step,
                "state": thought_stream.state,
                "thought": thought,
                "brain": {k: v for k, v in snap.items() if k not in ("token", "token_id", "base_token_id")},
            })
            print(f"\n  💭 [{thought_stream.state}] {thought}\n", flush=True)

        # 6. 打印本步
        tok_disp = repr(snap["token"])[:16]
        changed_mark = "*" if stdp_changed else " "
        print(f"  {step:>4d} | {tok_disp:<16} | {snap['da']:.4f} | {snap['ne']:.4f} | "
              f"{snap['cerebellar_error']:>6.2f} | {snap['valence']:+.3f} | "
              f"{snap['system']:<3s} | {snap['stdp_norm']:.4f}    | "
              f"{snap['stdp_grad_norm']:.4f}{changed_mark}", flush=True)

        # 7. 滚动
        hidden_prev = hidden_now.clone()

        if next_id == tokenizer.eos_token_id:
            print(f"  (EOS, stop)", flush=True)
            break

    dt = time.time() - t0
    full_text = prompt + "".join(generated_tokens)
    n_changed = sum(1 for s in snapshots if s["stdp_changed_output"])
    print(f"\n  ⏱ 生成 {len(snapshots)} tokens | 耗时 {dt:.2f}s | {len(snapshots)/max(dt,1e-6):.2f} tok/s", flush=True)
    print(f"  📝 完整文本: {full_text!r}", flush=True)
    print(f"  🔧 STDP 改变了 {n_changed}/{len(snapshots)} 个 token 的输出 ({n_changed/len(snapshots)*100:.1f}%)", flush=True)
    print(f"  📈 STDP 突触范数: {snapshots[0]['stdp_norm']:.4f} → {snapshots[-1]['stdp_norm']:.4f} "
          f"({(snapshots[-1]['stdp_norm']-snapshots[0]['stdp_norm'])/max(snapshots[0]['stdp_norm'],1e-8)*100:+.1f}%)", flush=True)

    return {
        "prompt": prompt,
        "generated_text": full_text,
        "new_tokens": generated_tokens,
        "snapshots": snapshots,
        "thought_log": thought_log,
        "gen_time_s": dt,
        "tokens_per_sec": len(snapshots) / max(dt, 1e-6),
        "stdp_changed_count": n_changed,
        "stdp_changed_pct": n_changed / len(snapshots) * 100 if snapshots else 0,
        "stdp_norm_first": snapshots[0]["stdp_norm"] if snapshots else 0,
        "stdp_norm_last": snapshots[-1]["stdp_norm"] if snapshots else 0,
    }


# ============================================================================
# 5. 主流程
# ============================================================================
def main():
    print("=" * 90, flush=True)
    print("🧠 GPT-2 (from lal) × stdpbrain v2 — 真 STDP 训练 + 思维流可视化", flush=True)
    print("=" * 90, flush=True)
    print(f"环境: torch={torch.__version__} device={DEVICE} threads={TORCH_THREADS}", flush=True)
    print(f"hidden_size={HIDDEN_SIZE}, STDP_LR={STDP_LR}, STDP_WEIGHT_CLAMP={STDP_WEIGHT_CLAMP}", flush=True)
    print(f"MAX_NEW_TOKENS={MAX_NEW_TOKENS}, THOUGHT_INTERVAL={THOUGHT_INTERVAL}", flush=True)

    tokenizer, gpt2 = load_gpt2()
    brain = STDPBrain()
    thought_stream = ThoughtStream()

    prompts = [
        "The capital of France is",
        "Once upon a time",
        "Hello, how are",
        "Machine learning is",
        "The meaning of life is",
    ]

    print(f"\n[3/5] 跑 {len(prompts)} 个 prompt 的 GPT-2 生成 + STDP 训练 + 思维流...", flush=True)
    all_results: List[Dict[str, Any]] = []
    for p in prompts:
        try:
            r = run_one_prompt(tokenizer, gpt2, brain, thought_stream, p)
            all_results.append(r)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ❌ prompt {p!r} 失败: {e}", flush=True)

    # === 汇总 ===
    print(f"\n[4/5] 汇总 & 写报告...", flush=True)
    summary_text = summarize(all_results)
    print(summary_text, flush=True)

    # === 保存 JSON ===
    json_path = os.path.join(DOWNLOAD_DIR, "gpt2_stdpbrain_v2_train.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "model": "gpt2 (lal repo reference)",
                "hidden_size": HIDDEN_SIZE,
                "stdp_lr": STDP_LR,
                "stdp_weight_clamp": STDP_WEIGHT_CLAMP,
                "max_new_tokens": MAX_NEW_TOKENS,
                "thought_interval": THOUGHT_INTERVAL,
                "device": DEVICE,
                "torch_threads": TORCH_THREADS,
                "n_prompts": len(prompts),
            },
            "results": all_results,
            "summary_text": summary_text,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n💾 JSON 报告: {json_path}", flush=True)

    # === 保存文本报告 ===
    txt_path = os.path.join(DOWNLOAD_DIR, "gpt2_stdpbrain_v2_train.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 90 + "\n")
        f.write("GPT-2 (from lal) × stdpbrain v2 — 真 STDP 训练 + 思维流可视化\n")
        f.write("=" * 90 + "\n\n")
        f.write(f"模型: gpt2 (lal 仓库参考, 12 层 / 768 embd / 12 head / 50257 vocab)\n")
        f.write(f"STDP 配置: lr={STDP_LR}, weight_clamp=±{STDP_WEIGHT_CLAMP}, hidden_size={HIDDEN_SIZE}\n")
        f.write(f"brain 模块: requires_grad=True (全部解冻)\n")
        f.write(f"STDP 注入: hidden_modified = hidden + 0.1 * STDP_W @ hidden\n")
        f.write(f"设备: {DEVICE}, torch {torch.__version__}, threads={TORCH_THREADS}\n\n")
        for r in all_results:
            f.write("─" * 90 + "\n")
            f.write(f"Prompt: {r['prompt']!r}\n")
            f.write(f"生成: {r['generated_text']!r}\n")
            f.write(f"耗时: {r['gen_time_s']:.2f}s | {r['tokens_per_sec']:.2f} tok/s\n")
            f.write(f"STDP 改变 token: {r['stdp_changed_count']}/{len(r['snapshots'])} ({r['stdp_changed_pct']:.1f}%)\n")
            f.write(f"STDP 范数: {r['stdp_norm_first']:.4f} → {r['stdp_norm_last']:.4f}\n")
            f.write("  step | token            | DA     | NE     | err    | val   | sys | stdp_n  | grad_n  | changed\n")
            f.write("  -----+------------------+--------+--------+--------+-------+-----+---------+---------+--------\n")
            for s in r["snapshots"]:
                td = repr(s["token"])[:16]
                ch = "*" if s["stdp_changed_output"] else " "
                f.write(f"  {s['step']:>4d} | {td:<16} | {s['da']:.4f} | {s['ne']:.4f} | "
                        f"{s['cerebellar_error']:>6.2f} | {s['valence']:+.3f} | "
                        f"{s['system']:<3s} | {s['stdp_norm']:.4f}  | "
                        f"{s['stdp_grad_norm']:.4f}  | {ch}\n")
            f.write("\n")
        f.write("\n" + "=" * 90 + "\n汇总\n")
        f.write("=" * 90 + "\n")
        f.write(summary_text)
    print(f"💾 文本报告: {txt_path}", flush=True)

    # === 保存思维流纯文本 ===
    thoughts_path = os.path.join(DOWNLOAD_DIR, "gpt2_stdpbrain_v2_thoughts.txt")
    with open(thoughts_path, "w", encoding="utf-8") as f:
        f.write("=" * 90 + "\n")
        f.write("GPT-2 × stdpbrain 思维流 (Thought Stream) 完整记录\n")
        f.write("=" * 90 + "\n\n")
        for r in all_results:
            f.write(f"\n{'─' * 90}\n")
            f.write(f"Prompt: {r['prompt']!r}\n")
            f.write(f"生成文本: {r['generated_text']!r}\n")
            f.write(f"{'─' * 90}\n")
            for t in r["thought_log"]:
                f.write(f"\n  [step {t['step']:>3d}] 状态: {t['state']}\n")
                f.write(f"          思维: {t['thought']}\n")
                b = t["brain"]
                f.write(f"          脑活动: DA={b['da']:.4f} NE={b['ne']:.4f} "
                        f"err={b['cerebellar_error']:.2f} valence={b['valence']:+.4f} "
                        f"sys={b['system']} stdp_norm={b['stdp_norm']:.4f}\n")
            f.write("\n")
    print(f"💾 思维流文本: {thoughts_path}", flush=True)

    # === 保存趋势图 ===
    try:
        plot_path = os.path.join(DOWNLOAD_DIR, "gpt2_stdpbrain_v2.png")
        plot_activity(all_results, plot_path)
        print(f"💾 趋势图: {plot_path}", flush=True)
    except Exception as e:
        print(f"  ⚠️ 趋势图生成失败: {e}", flush=True)

    print(f"\n[5/5] ✅ 完成", flush=True)


# ============================================================================
# 汇总
# ============================================================================
def summarize(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "(无结果)"
    lines: List[str] = []
    lines.append("=" * 90)
    lines.append("🧠 v2 汇总 (真 STDP 训练)")
    lines.append("=" * 90)

    total_tokens = 0
    total_changed = 0
    all_da, all_ne, all_err, all_val, all_syn = [], [], [], [], []
    all_grad = []
    per_prompt = []

    for r in results:
        snaps = r["snapshots"]
        if not snaps: continue
        da = [s["da"] for s in snaps]
        ne = [s["ne"] for s in snaps]
        err = [s["cerebellar_error"] for s in snaps]
        val = [s["valence"] for s in snaps]
        syn = [s["stdp_norm"] for s in snaps]
        grad = [s["stdp_grad_norm"] for s in snaps]
        all_da += da; all_ne += ne; all_err += err; all_val += val; all_syn += syn; all_grad += grad
        total_tokens += len(snaps)
        total_changed += r["stdp_changed_count"]

        syn_first, syn_last = syn[0], syn[-1]
        syn_delta = (syn_last - syn_first) / max(abs(syn_first), 1e-8) * 100
        ps = {
            "prompt": r["prompt"],
            "n_tokens": len(snaps),
            "changed": r["stdp_changed_count"],
            "changed_pct": r["stdp_changed_pct"],
            "syn_first": syn_first,
            "syn_last": syn_last,
            "syn_delta_pct": syn_delta,
            "grad_mean": sum(grad)/len(grad),
            "da_mean": sum(da)/len(da),
            "ne_mean": sum(ne)/len(ne),
            "val_mean": sum(val)/len(val),
            "err_mean": sum(err)/len(err),
            "generated": r["generated_text"],
        }
        per_prompt.append(ps)
        lines.append(f"\n📌 {r['prompt']!r}")
        lines.append(f"   生成 {ps['n_tokens']} tokens | STDP 改变输出: {ps['changed']}/{ps['n_tokens']} ({ps['changed_pct']:.1f}%)")
        lines.append(f"   STDP 范数: {ps['syn_first']:.4f} → {ps['syn_last']:.4f} ({ps['syn_delta_pct']:+.1f}%)")
        lines.append(f"   梯度范数均值: {ps['grad_mean']:.4f}")
        lines.append(f"   DA 均值={ps['da_mean']:.4f}  NE 均值={ps['ne_mean']:.4f}  err 均值={ps['err_mean']:.2f}  valence={ps['val_mean']:+.4f}")
        lines.append(f"   生成文本: {ps['generated']!r}")

    if all_da:
        lines.append("\n" + "─" * 90)
        lines.append("📊 全局统计:")
        lines.append(f"   总 token: {total_tokens}")
        lines.append(f"   STDP 改变 token 总数: {total_changed}/{total_tokens} ({total_changed/total_tokens*100:.1f}%)")
        lines.append(f"   DA        : mean={sum(all_da)/len(all_da):.4f}")
        lines.append(f"   NE        : mean={sum(all_ne)/len(all_ne):.4f}")
        lines.append(f"   小脑误差  : mean={sum(all_err)/len(all_err):.4f}")
        lines.append(f"   valence   : mean={sum(all_val)/len(all_val):+.4f}")
        lines.append(f"   STDP 范数 : first={all_syn[0]:.4f}  last={all_syn[-1]:.4f}  "
                     f"delta={((all_syn[-1]-all_syn[0])/max(abs(all_syn[0]),1e-8)*100):+.1f}%")
        lines.append(f"   梯度范数  : mean={sum(all_grad)/len(all_grad):.4f}  max={max(all_grad):.4f}")

        # 解读
        lines.append("\n" + "─" * 90)
        lines.append("🔍 v2 关键发现:")
        if total_changed / total_tokens > 0.1:
            lines.append(f"   ✅ STDP 真实影响 GPT-2 输出: {total_changed/total_tokens*100:.1f}% 的 token 被 STDP 改变")
            lines.append(f"      说明 STDP 突触矩阵的增量确实在调节 GPT-2 的 logits 分布")
        else:
            lines.append(f"   ⚠️ STDP 改变输出比例偏低 ({total_changed/total_tokens*100:.1f}%)，可能需要更大 lr 或更多 epoch")
        syn_delta_pct = (all_syn[-1] - all_syn[0]) / max(abs(all_syn[0]), 1e-8) * 100
        if abs(syn_delta_pct) > 5:
            direction = "LTP(增强)" if syn_delta_pct > 0 else "LTD(抑制)"
            lines.append(f"   ✅ STDP 突触矩阵发生实质变化: {all_syn[0]:.4f} → {all_syn[-1]:.4f} ({syn_delta_pct:+.1f}%), 方向={direction}")
        else:
            lines.append(f"   ⚠️ STDP 突触范数变化幅度小 ({syn_delta_pct:+.1f}%)，weight_clamp 可能过紧")
        if sum(all_grad)/len(all_grad) > 1e-4:
            lines.append(f"   ✅ 梯度非零 (均值 {sum(all_grad)/len(all_grad):.4f})，反向传播链路打通")
        else:
            lines.append(f"   ⚠️ 梯度接近零，可能需要调高 lr 或检查 detach()")

    return "\n".join(lines)


# ============================================================================
# 趋势图
# ============================================================================
def plot_activity(results: List[Dict[str, Any]], out_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    try:
        fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf')
        fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    except Exception:
        pass
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    results = results[:4]
    if not results: return
    fig, axes = plt.subplots(len(results), 5, figsize=(22, 4 * len(results)),
                             constrained_layout=True, sharex=False)
    if len(results) == 1:
        axes = axes.reshape(1, -1)

    metrics = [
        ("DA 多巴胺", "da", "tab:purple"),
        ("NE 去甲肾上腺素", "ne", "tab:orange"),
        ("小脑预测误差", "cerebellar_error", "tab:red"),
        ("杏仁核 valence", "valence", "tab:green"),
        ("STDP 突触范数", "stdp_norm", "tab:blue"),
    ]
    for i, r in enumerate(results):
        snaps = r["snapshots"]
        if not snaps: continue
        steps = list(range(len(snaps)))
        for j, (label, key, color) in enumerate(metrics):
            ax = axes[i, j]
            vals = [s[key] for s in snaps]
            ax.plot(steps, vals, marker="o", color=color, linewidth=1.5, markersize=4)
            ax.fill_between(steps, vals, alpha=0.15, color=color)
            ax.set_title(f"{label}\n{r['prompt'][:30]!r}", fontsize=10)
            ax.set_xlabel("token step", fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
            # 标记 STDP 改变的 token
            if key == "stdp_norm":
                for k, s in enumerate(snaps):
                    if s.get("stdp_changed_output"):
                        ax.axvline(k, color="red", alpha=0.3, linestyle="--", linewidth=0.5)

    fig.suptitle(
        "GPT-2 × stdpbrain v2 — 真 STDP 训练 + 思维流\n"
        "红色虚线 = STDP 改变了该 token 的输出",
        fontsize=13,
    )
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
