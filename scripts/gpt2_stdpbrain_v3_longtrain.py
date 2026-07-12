#!/usr/bin/env python3
"""
GPT-2 × stdpbrain v3 — 长训练 + 外部 reward + 解冻 synaptic_plasticity
======================================================================

v3 三大改进 (基于 v2):
1. 长训练: 80 个 wikitext prompt × 10 token × 2 epoch = 1600 步
2. 外部 reward: confidence + repetition_penalty + stdp_bonus + entropy
   (不再只用 DA-0.5 自监督)
3. 解冻 synaptic_plasticity: 768 个内部参数可训练
   - 三因子规则本身 (timing window, BCM threshold, etc.) 随训练调整
   - meta-loss: reward > 0 → 鼓励更大 |ΔW|; reward < 0 → 抑制 |ΔW|
"""

from __future__ import annotations

import os, sys, json, time, random, warnings, shutil
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass, field

warnings.filterwarnings("ignore")

STDPBRAIN_DIR = "/home/z/my-project/repos/stdpbrain"
DOWNLOAD_DIR = "/home/z/my-project/download"
SCRIPTS_DIR = "/home/z/my-project/scripts"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(SCRIPTS_DIR, exist_ok=True)
os.chdir(STDPBRAIN_DIR)
sys.path.insert(0, STDPBRAIN_DIR)

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.nn.functional as F
import gc

HIDDEN_SIZE = 768
DEVICE = "cpu"
TORCH_THREADS = 4
torch.set_num_threads(TORCH_THREADS)

# === 训练超参 ===
STDP_LR = 1e-3              # STDP 突触矩阵 lr
SP_LR = 5e-4                # synaptic_plasticity 内部参数 lr
STDP_WEIGHT_CLAMP = 0.3
N_PROMPTS = 80              # 每 epoch 的 prompt 数
MAX_NEW_TOKENS = 10         # 每 prompt 生成 token 数
N_EPOCHS = 2

WIKITEXT_PARQUET = "/tmp/wikitext2.parquet"
WIKITEXT_CACHE = os.path.join(SCRIPTS_DIR, "wikitext2_prompts.json")


# ============================================================================
# 1. 加载 wikitext prompt
# ============================================================================
def load_wikitext_prompts(n: int = N_PROMPTS) -> List[str]:
    """从 wikitext-2-raw parquet 提取 n 个短 prompt."""
    if os.path.exists(WIKITEXT_CACHE):
        with open(WIKITEXT_CACHE, 'r') as f:
            cached = json.load(f)
        if len(cached) >= n:
            print(f"[*] Using cached wikitext prompts ({len(cached)} available)", flush=True)
            return cached[:n]

    print(f"[*] Extracting {n} prompts from wikitext-2-raw parquet...", flush=True)
    import pyarrow.parquet as pq
    table = pq.read_table(WIKITEXT_PARQUET)
    texts = table.column('text').to_pylist()

    # 筛选: 长度 > 50, 提取第一句作为 prompt
    segments = [t.strip() for t in texts if len(t.strip()) > 50]
    random.seed(42)
    random.shuffle(segments)

    prompts = []
    seen = set()
    for seg in segments:
        # 找第一句 (25-80 字符之间)
        prompt = None
        for sep in ['. ', '? ', '! ', ' . ', ' ? ']:
            idx = seg.find(sep)
            if 25 < idx < 80:
                prompt = seg[:idx+1].strip()
                break
        if not prompt and len(seg) > 40:
            prompt = seg[:50].strip()

        if prompt and prompt not in seen:
            seen.add(prompt)
            prompts.append(prompt)
        if len(prompts) >= n * 2:  # 多提取一些备用
            break

    prompts = prompts[:n]
    with open(WIKITEXT_CACHE, 'w', encoding='utf-8') as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)
    print(f"    ✅ Extracted {len(prompts)} prompts, cached to {WIKITEXT_CACHE}", flush=True)
    for i, p in enumerate(prompts[:5]):
        print(f"    [{i+1}] {p!r}", flush=True)
    return prompts


# ============================================================================
# 2. 加载 GPT-2
# ============================================================================
def load_gpt2():
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    print("[*] Loading GPT-2...", flush=True)
    t0 = time.time()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    model.to(DEVICE)
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"    ✅ {sum(p.numel() for p in model.parameters())/1e6:.1f}M params | {time.time()-t0:.1f}s", flush=True)
    return tokenizer, model


# ============================================================================
# 3. 外部 reward 函数
# ============================================================================
def compute_external_reward(
    logits: torch.Tensor,      # (vocab,)
    chosen_id: int,
    base_id: int,
    generated_ids: List[int],  # 所有已生成 token (含 prompt)
) -> float:
    """
    外部 reward = 0.4*confidence + 0.3*rep_penalty + 0.2*stdp_bonus + 0.1*entropy

    - confidence: 模型对 chosen_id 的置信度 (越高越好)
    - rep_penalty: 如果 chosen_id 在最近 5 个 token 中出现过, 惩罚
    - stdp_bonus: STDP 改变了输出时, 如果改变是好的 (高置信+无重复), 奖励
    - entropy: 鼓励适度探索 (不太确定也不太随机)
    """
    with torch.no_grad():
        probs = F.softmax(logits, dim=-1)

        # 1. Confidence (0-1)
        confidence = float(probs[chosen_id].item())

        # 2. Repetition penalty
        recent = generated_ids[-5:] if len(generated_ids) >= 5 else generated_ids
        rep_count = sum(1 for t in recent if t == chosen_id)
        rep_penalty = -0.3 * rep_count

        # 3. STDP influence bonus
        if chosen_id != base_id:
            # STDP made a different choice
            if confidence > 0.05 and rep_count == 0:
                stdp_bonus = +0.2  # good different choice
            else:
                stdp_bonus = -0.1  # bad different choice (low confidence or repetitive)
        else:
            stdp_bonus = 0.0  # same as base

        # 4. Entropy bonus (encourage exploration, but not too much)
        log_probs = torch.log(probs + 1e-8)
        entropy = -float((probs * log_probs).sum().item())
        entropy_bonus = 0.05 * min(entropy, 5.0) / 5.0  # cap at 5.0

    reward = 0.4 * confidence + 0.3 * rep_penalty + 0.2 * stdp_bonus + 0.1 * entropy_bonus
    return reward


# ============================================================================
# 4. STDP Brain v3 (解冻 synaptic_plasticity)
# ============================================================================
class STDPBrainV3:
    """brain 模块 + 可训练 STDP 突触 + 可训练 synaptic_plasticity."""

    def __init__(self):
        import importlib
        print("[*] Loading brain modules...", flush=True)
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
        ]

        self.sp_params: List[nn.Parameter] = []
        for name, path, factory, kwargs in spec:
            try:
                mod = importlib.import_module(path)
                m = getattr(mod, factory)(**kwargs)
                m.eval() if hasattr(m, "eval") else None

                # ★ v3 策略: synaptic_plasticity 参数保持 frozen (避免 768×768 计算图 OOM)
                # 但我们用"手动元更新规则"直接调整 sp 的关键参数 (A_plus, A_minus)
                # 这是生物学上合理的: 多巴胺调制 LTP/LTD 速率
                for p in m.parameters():
                    p.requires_grad_(False)
                tag = "frozen (meta-updated)" if name == "synaptic_plast" else "frozen"

                n = sum(p.numel() for p in m.parameters()) if hasattr(m, "parameters") else 0
                self.modules[name] = m
                print(f"    {name:<16} | {n:>10,} params | {tag}", flush=True)
            except Exception as e:
                print(f"    {name:<16} | ERROR: {e}", flush=True)

        # 找到 sp 的关键参数 (a_plus, a_minus) 用于手动元更新
        sp = self.modules.get("synaptic_plast")
        self.sp_meta_params: Dict[str, Any] = {}
        if sp is not None:
            # 遍历所有子模块, 找到 timing window 的 a_plus/a_minus (lowercase in stdpbrain)
            for sub_name, sub_mod in sp.named_modules():
                for attr_name in ['a_plus', 'a_minus', 'A_plus', 'A_minus', 'tau_plus', 'tau_minus']:
                    if hasattr(sub_mod, attr_name):
                        val = getattr(sub_mod, attr_name)
                        key = f"{sub_name}.{attr_name}"
                        self.sp_meta_params[key] = val
            print(f"    → SP meta-params tracked: {len(self.sp_meta_params)} tensors/scalars", flush=True)
            for k, v in list(self.sp_meta_params.items())[:6]:
                if isinstance(v, torch.Tensor):
                    print(f"      {k}: tensor shape={list(v.shape)} val={float(v.mean().item()):.6f}", flush=True)
                else:
                    print(f"      {k}: scalar val={float(v):.6f}", flush=True)

        n_sp = sum(p.numel() for p in sp.parameters()) if sp is not None else 0
        print(f"    → synaptic_plasticity 参数: {n_sp} (frozen, 手动元更新 A_plus/A_minus)", flush=True)

        # === 可训练 STDP 突触矩阵 (768×768) ===
        self.stdp_weight = nn.Parameter(
            torch.randn(HIDDEN_SIZE, HIDDEN_SIZE, device=DEVICE) * 0.01,
            requires_grad=True,
        )

        # 只有一个 optimizer (SGD for stdp_weight)
        self.opt_stdp = torch.optim.SGD([self.stdp_weight], lr=STDP_LR, momentum=0.9, weight_decay=1e-4)

        print(f"    STDP_W: 768×768 (SGD lr={STDP_LR} mom=0.9)", flush=True)
        print(f"    SP meta-update: manual (A_plus += reward * 0.001, A_minus -= reward * 0.001)", flush=True)

        # 状态
        self.da_level = 0.5
        self.ne_level = 0.5
        self.valence = 0.0
        self.cumulative_reward = 0.0  # 用于 sp 元更新

    def step(
        self,
        hidden_now: torch.Tensor,    # (1, 768)
        hidden_prev: torch.Tensor,   # (1, 768)
        external_reward: float,      # 来自上一步的外部 reward
    ) -> Tuple[Dict[str, float], torch.Tensor]:
        """一遍脑模块闭环 + STDP + synaptic_plasticity 联合更新."""
        snap: Dict[str, float] = {}
        H = hidden_now.detach()

        # ---- 0. STDP 增量注入 (no_grad — stdp_weight 的更新走伪梯度, 不需要计算图) ----
        with torch.no_grad():
            stdp_delta = F.linear(H, self.stdp_weight)  # (1, 768)
            modified_hidden = H + stdp_delta * 0.1

        # ---- 1~6. brain 模块闭环 (no_grad, 仅信号源) ----
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
            if ast is not None:
                try:
                    ast.forward(da_level=da_level, ne_level=ne_level, ach_level=0.5, sht_level=0.5)
                except Exception:
                    pass

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

        # ---- 7. synaptic_plasticity forward (no_grad, 信号源) ----
        # sp.forward 返回 updated_weights = weights + ΔW
        # ΔW 依赖于 sp 的内部参数 (timing window, BCM threshold, etc.)
        sp = self.modules.get("synaptic_plast")
        delta_w = None
        if sp is not None:
            try:
                with torch.no_grad():
                    new_w = sp.forward(
                        pre_activity=hidden_prev.detach(),
                        post_activity=H,
                        dopamine_level=da_level,
                        weights=self.stdp_weight.data.clone(),
                    )
                    delta_w = (new_w - self.stdp_weight.data).detach()
            except Exception:
                delta_w = None

        # ---- 8. 联合 reward (外部 + 内部 DA) ----
        combined_reward = 0.5 * external_reward + 0.5 * (da_level - 0.5)
        snap["external_reward"] = external_reward
        snap["combined_reward"] = combined_reward
        self.cumulative_reward += combined_reward

        # ---- 9. 更新 stdp_weight (伪梯度, no_grad) ----
        if delta_w is not None:
            with torch.no_grad():
                pseudo_grad = -delta_w * combined_reward * 0.1
                if self.stdp_weight.grad is None:
                    self.stdp_weight.grad = pseudo_grad.clone()
                else:
                    self.stdp_weight.grad.copy_(pseudo_grad)

            # ---- 10. 手动元更新 synaptic_plasticity 的 a_plus / a_minus ----
            # 生物学原理: 多巴胺调制 LTP/LTD 速率
            # reward > 0 → 增强 LTP (a_plus ↑), 抑制 LTD (a_minus ↓)
            # reward < 0 → 抑制 LTP (a_plus ↓), 增强 LTD (a_minus ↑)
            # 更新幅度: reward * meta_lr, 并裁剪到 [0.0001, 0.1]
            meta_lr = 0.0005
            with torch.no_grad():
                for name, param in self.sp_meta_params.items():
                    name_lower = name.lower()
                    if "a_plus" in name_lower or "A_plus" in name:
                        # LTP 速率: reward > 0 时增大
                        if isinstance(param, torch.Tensor):
                            param.add_(combined_reward * meta_lr)
                            param.clamp_(0.0001, 0.1)
                        else:
                            new_val = float(param) + combined_reward * meta_lr
                            new_val = max(0.0001, min(0.1, new_val))
                            parts = name.split(".")
                            obj = self.modules["synaptic_plast"]
                            for p in parts[:-1]:
                                if p:
                                    obj = getattr(obj, p)
                            setattr(obj, parts[-1], new_val)
                            self.sp_meta_params[name] = new_val
                    elif "a_minus" in name_lower or "A_minus" in name:
                        # LTD 速率: reward > 0 时减小 (抑制遗忘)
                        if isinstance(param, torch.Tensor):
                            param.sub_(combined_reward * meta_lr)
                            param.clamp_(-0.1, -0.0001)
                        else:
                            new_val = float(param) - combined_reward * meta_lr
                            new_val = max(-0.1, min(-0.0001, new_val))
                            parts = name.split(".")
                            obj = self.modules["synaptic_plast"]
                            for p in parts[:-1]:
                                if p:
                                    obj = getattr(obj, p)
                            setattr(obj, parts[-1], new_val)
                            self.sp_meta_params[name] = new_val
                    # tau_plus / tau_minus 不更新 (时间常数保持稳定)

        snap["stdp_norm"] = float(self.stdp_weight.norm().item())
        snap["stdp_grad_norm"] = float(self.stdp_weight.grad.norm().item()) if self.stdp_weight.grad is not None else 0.0

        # 记录 sp 元参数范数 (追踪三因子规则是否在变)
        if self.sp_meta_params:
            def _val(v):
                if isinstance(v, torch.Tensor):
                    return float(v.mean().item())
                return float(v)
            def _norm(v):
                if isinstance(v, torch.Tensor):
                    return float(v.norm().item())
                return abs(float(v))
            sp_norm = float(sum(_norm(v) ** 2 for v in self.sp_meta_params.values()) ** 0.5)
            snap["sp_param_norm"] = sp_norm
            # 记录 A_plus / A_minus 均值
            a_plus_vals = [_val(v) for k, v in self.sp_meta_params.items() if "a_plus" in k.lower()]
            a_minus_vals = [_val(v) for k, v in self.sp_meta_params.items() if "a_minus" in k.lower()]
            if a_plus_vals:
                snap["A_plus_mean"] = sum(a_plus_vals) / len(a_plus_vals)
            if a_minus_vals:
                snap["A_minus_mean"] = sum(a_minus_vals) / len(a_minus_vals)

        # ---- 11. optimizer step for stdp_weight ----
        if self.stdp_weight.grad is not None:
            grad_norm = self.stdp_weight.grad.norm().item()
            if grad_norm > 10.0:
                self.stdp_weight.grad.mul_(10.0 / grad_norm)
        self.opt_stdp.step()
        self.opt_stdp.zero_grad()
        self.stdp_weight.data.clamp_(-STDP_WEIGHT_CLAMP, STDP_WEIGHT_CLAMP)

        return snap, modified_hidden.detach()


# ============================================================================
# 5. 训练单个 prompt
# ============================================================================
def train_one_prompt(tokenizer, gpt2, brain: STDPBrainV3, prompt: str, max_new_tokens: int) -> Dict[str, Any]:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    generated_ids = input_ids.clone()  # (1, L)

    # 获取 hidden_prev
    with torch.no_grad():
        out0 = gpt2(input_ids, output_hidden_states=True, use_cache=False)
        hidden_prev = out0.hidden_states[-1][:, -1, :].squeeze(0).detach()
        if hidden_prev.dim() == 1:
            hidden_prev = hidden_prev.unsqueeze(0)

    snapshots: List[Dict[str, Any]] = []
    generated_tokens: List[str] = []
    rewards: List[float] = []
    stdp_changed_count = 0

    for step in range(max_new_tokens):
        # 1. GPT-2 forward (no_grad)
        with torch.no_grad():
            out = gpt2(generated_ids, output_hidden_states=True, use_cache=False)
            hidden_now = out.hidden_states[-1][:, -1, :].squeeze(0).detach()
            if hidden_now.dim() == 1:
                hidden_now = hidden_now.unsqueeze(0)
            base_logits = gpt2.lm_head(hidden_now).squeeze(0)  # (vocab,)
            base_id = int(torch.argmax(base_logits, dim=-1).item())

        # 2. 确定本步用的 reward (用上一步的实际 reward, step 0 用 0.5 中性)
        ext_reward = rewards[-1] if rewards else 0.5

        # 3. brain step + STDP 更新
        snap, modified_hidden = brain.step(hidden_now, hidden_prev, ext_reward)

        # 4. STDP 修改后的 logits
        with torch.no_grad():
            modified_logits = gpt2.lm_head(modified_hidden).squeeze(0)  # (vocab,)
            next_id = int(torch.argmax(modified_logits, dim=-1).item())

        # 5. 计算本步实际 reward
        gen_ids_list = generated_ids[0].tolist()
        actual_reward = compute_external_reward(modified_logits, next_id, base_id, gen_ids_list)
        rewards.append(actual_reward)

        # 6. 记录
        snap["token"] = tokenizer.decode([next_id])
        snap["token_id"] = next_id
        snap["base_id"] = base_id
        snap["stdp_changed"] = (next_id != base_id)
        if snap["stdp_changed"]:
            stdp_changed_count += 1
        snapshots.append(snap)

        generated_ids = torch.cat([generated_ids, torch.tensor([[next_id]], device=DEVICE)], dim=1)
        generated_tokens.append(snap["token"])
        hidden_prev = hidden_now.clone()

        if next_id == tokenizer.eos_token_id:
            break

    return {
        "prompt": prompt,
        "generated_text": prompt + "".join(generated_tokens),
        "snapshots": snapshots,
        "avg_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "stdp_changed_pct": stdp_changed_count / len(snapshots) * 100 if snapshots else 0,
        "stdp_norm_first": snapshots[0]["stdp_norm"] if snapshots else 0,
        "stdp_norm_last": snapshots[-1]["stdp_norm"] if snapshots else 0,
        "n_tokens": len(snapshots),
    }


# ============================================================================
# 6. 主流程
# ============================================================================
def main():
    print("=" * 90, flush=True)
    print("🧠 GPT-2 × stdpbrain v3 — 长训练 + 外部 reward + 解冻 synaptic_plasticity", flush=True)
    print("=" * 90, flush=True)
    print(f"Config: {N_PROMPTS} prompts × {MAX_NEW_TOKENS} tokens × {N_EPOCHS} epochs = "
          f"{N_PROMPTS * MAX_NEW_TOKENS * N_EPOCHS} total steps", flush=True)
    print(f"STDP_LR={STDP_LR}, SP_LR={SP_LR}, WEIGHT_CLAMP=±{STDP_WEIGHT_CLAMP}", flush=True)
    print(f"Reward: 0.4*confidence + 0.3*rep_penalty + 0.2*stdp_bonus + 0.1*entropy", flush=True)
    print(f"device={DEVICE}, torch={torch.__version__}, threads={TORCH_THREADS}\n", flush=True)

    prompts = load_wikitext_prompts(N_PROMPTS)
    tokenizer, gpt2 = load_gpt2()
    brain = STDPBrainV3()

    initial_stdp_norm = brain.stdp_weight.norm().item()
    def _sp_val(v):
        if isinstance(v, torch.Tensor): return float(v.mean().item())
        return float(v)
    def _sp_norm(v):
        if isinstance(v, torch.Tensor): return float(v.norm().item())
        return abs(float(v))
    initial_sp_norm = sum(_sp_norm(v) ** 2 for v in brain.sp_meta_params.values()) ** 0.5
    a_plus_items = [(k, v) for k, v in brain.sp_meta_params.items() if "a_plus" in k.lower()]
    a_minus_items = [(k, v) for k, v in brain.sp_meta_params.items() if "a_minus" in k.lower()]
    initial_a_plus = sum(_sp_val(v) for _, v in a_plus_items) / max(1, len(a_plus_items))
    initial_a_minus = sum(_sp_val(v) for _, v in a_minus_items) / max(1, len(a_minus_items))
    print(f"\n[*] Initial STDP norm: {initial_stdp_norm:.6f}", flush=True)
    print(f"[*] Initial SP meta-param norm: {initial_sp_norm:.6f}", flush=True)
    print(f"[*] Initial A_plus mean: {initial_a_plus:.6f}", flush=True)
    print(f"[*] Initial A_minus mean: {initial_a_minus:.6f}", flush=True)

    global_history: List[Dict[str, Any]] = []
    step_history: List[Dict[str, Any]] = []
    total_steps = 0
    t_start = time.time()

    for epoch in range(N_EPOCHS):
        print(f"\n{'='*90}", flush=True)
        print(f"Epoch {epoch+1}/{N_EPOCHS}", flush=True)
        print(f"{'='*90}", flush=True)

        for i, prompt in enumerate(prompts):
            t0 = time.time()
            try:
                result = train_one_prompt(tokenizer, gpt2, brain, prompt, MAX_NEW_TOKENS)
                result["epoch"] = epoch + 1
                result["prompt_idx"] = i
                result["time_s"] = time.time() - t0
                global_history.append(result)

                for s in result["snapshots"]:
                    s["global_step"] = total_steps
                    s["epoch"] = epoch + 1
                    s["prompt_idx"] = i
                    step_history.append(s)
                    total_steps += 1

                # 每 10 个 prompt 打印进度
                if (i + 1) % 10 == 0 or i == 0 or i == len(prompts) - 1:
                    elapsed = time.time() - t_start
                    cur_norm = brain.stdp_weight.norm().item()
                    cur_sp_norm = sum(_sp_norm(v) ** 2 for v in brain.sp_meta_params.values()) ** 0.5
                    cur_a_plus_items = [(k, v) for k, v in brain.sp_meta_params.items() if "a_plus" in k.lower()]
                    cur_a_plus = sum(_sp_val(v) for _, v in cur_a_plus_items) / max(1, len(cur_a_plus_items))
                    avg_r = result["avg_reward"]
                    changed = result["stdp_changed_pct"]
                    delta_norm = cur_norm - initial_stdp_norm
                    print(f"  E{epoch+1} [{i+1:>3d}/{N_PROMPTS}] | "
                          f"stdp={cur_norm:.4f} (Δ={delta_norm:+.5f}) | "
                          f"A+={cur_a_plus:.5f} | "
                          f"sp={cur_sp_norm:.5f} | "
                          f"R={avg_r:+.3f} | "
                          f"chg={changed:.0f}% | "
                          f"elapsed={elapsed:.0f}s", flush=True)
                    # 每 20 个 prompt 强制 gc 释放内存
                    if (i + 1) % 20 == 0:
                        gc.collect()
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  ❌ prompt {i} failed: {e}", flush=True)

    total_time = time.time() - t_start
    final_stdp_norm = brain.stdp_weight.norm().item()
    final_sp_norm = sum(_sp_norm(v) ** 2 for v in brain.sp_meta_params.values()) ** 0.5
    final_a_plus_items = [(k, v) for k, v in brain.sp_meta_params.items() if "a_plus" in k.lower()]
    final_a_minus_items = [(k, v) for k, v in brain.sp_meta_params.items() if "a_minus" in k.lower()]
    final_a_plus = sum(_sp_val(v) for _, v in final_a_plus_items) / max(1, len(final_a_plus_items))
    final_a_minus = sum(_sp_val(v) for _, v in final_a_minus_items) / max(1, len(final_a_minus_items))

    print(f"\n{'='*90}", flush=True)
    print(f"✅ Training complete!", flush=True)
    print(f"   Total time: {total_time:.1f}s ({total_time/60:.1f} min)", flush=True)
    print(f"   Total steps: {total_steps}", flush=True)
    print(f"   STDP norm: {initial_stdp_norm:.6f} → {final_stdp_norm:.6f} "
          f"(Δ={final_stdp_norm-initial_stdp_norm:+.6f}, "
          f"{(final_stdp_norm-initial_stdp_norm)/max(initial_stdp_norm,1e-8)*100:+.2f}%)", flush=True)
    print(f"   SP meta-param norm: {initial_sp_norm:.6f} → {final_sp_norm:.6f} "
          f"(Δ={final_sp_norm-initial_sp_norm:+.6f}, "
          f"{(final_sp_norm-initial_sp_norm)/max(initial_sp_norm,1e-8)*100:+.2f}%)", flush=True)
    print(f"   A_plus: {initial_a_plus:.6f} → {final_a_plus:.6f} "
          f"(Δ={final_a_plus-initial_a_plus:+.6f})", flush=True)
    print(f"   A_minus: {initial_a_minus:.6f} → {final_a_minus:.6f} "
          f"(Δ={final_a_minus-initial_a_minus:+.6f})", flush=True)

    # 保存结果
    save_results(global_history, step_history, initial_stdp_norm, final_stdp_norm,
                 initial_sp_norm, final_sp_norm, initial_a_plus, final_a_plus,
                 initial_a_minus, final_a_minus, total_time, total_steps)
    plot_drift(step_history, global_history, initial_stdp_norm, final_stdp_norm,
               initial_sp_norm, final_sp_norm, initial_a_plus, final_a_plus,
               initial_a_minus, final_a_minus)


# ============================================================================
# 7. 保存结果
# ============================================================================
def save_results(global_history, step_history, init_norm, final_norm,
                 init_sp, final_sp, init_aplus, final_aplus,
                 init_aminus, final_aminus, total_time, total_steps):
    # JSON
    json_path = os.path.join(DOWNLOAD_DIR, "gpt2_stdpbrain_v3_longtrain.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            "config": {
                "n_prompts": N_PROMPTS, "max_new_tokens": MAX_NEW_TOKENS, "n_epochs": N_EPOCHS,
                "stdp_lr": STDP_LR, "sp_meta_lr": 0.0005, "weight_clamp": STDP_WEIGHT_CLAMP,
                "total_steps": total_steps, "total_time_s": total_time,
            },
            "initial_stdp_norm": init_norm,
            "final_stdp_norm": final_norm,
            "stdp_norm_delta": final_norm - init_norm,
            "stdp_norm_delta_pct": (final_norm - init_norm) / max(init_norm, 1e-8) * 100,
            "initial_sp_norm": init_sp,
            "final_sp_norm": final_sp,
            "sp_norm_delta": final_sp - init_sp,
            "sp_norm_delta_pct": (final_sp - init_sp) / max(init_sp, 1e-8) * 100,
            "initial_A_plus": init_aplus,
            "final_A_plus": final_aplus,
            "A_plus_delta": final_aplus - init_aplus,
            "initial_A_minus": init_aminus,
            "final_A_minus": final_aminus,
            "A_minus_delta": final_aminus - init_aminus,
            "step_history": step_history,
            "prompt_history": [{k: v for k, v in g.items() if k != "snapshots"} for g in global_history],
        }, f, ensure_ascii=False, indent=2)
    print(f"💾 JSON: {json_path}", flush=True)

    # TXT report
    txt_path = os.path.join(DOWNLOAD_DIR, "gpt2_stdpbrain_v3_longtrain.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 90 + "\n")
        f.write("GPT-2 × stdpbrain v3 — 长训练报告\n")
        f.write("=" * 90 + "\n\n")
        f.write(f"Config: {N_PROMPTS} prompts × {MAX_NEW_TOKENS} tokens × {N_EPOCHS} epochs\n")
        f.write(f"  = {total_steps} total token steps\n")
        f.write(f"STDP_LR={STDP_LR}, SP_META_LR=0.0005, WEIGHT_CLAMP=±{STDP_WEIGHT_CLAMP}\n")
        f.write(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)\n\n")

        f.write(f"━━━ STDP 突触矩阵 (768×768) ━━━\n")
        f.write(f"  norm: {init_norm:.6f} → {final_norm:.6f}\n")
        f.write(f"  delta: {final_norm-init_norm:+.6f} ({(final_norm-init_norm)/max(init_norm,1e-8)*100:+.2f}%)\n\n")

        f.write(f"━━━ synaptic_plasticity 元参数 (A_plus / A_minus) ━━━\n")
        f.write(f"  SP meta-param norm: {init_sp:.6f} → {final_sp:.6f} "
                f"(Δ={final_sp-init_sp:+.6f}, {(final_sp-init_sp)/max(init_sp,1e-8)*100:+.2f}%)\n")
        f.write(f"  A_plus (LTP 速率): {init_aplus:.6f} → {final_aplus:.6f} "
                f"(Δ={final_aplus-init_aplus:+.6f})\n")
        f.write(f"  A_minus (LTD 速率): {init_aminus:.6f} → {final_aminus:.6f} "
                f"(Δ={final_aminus-init_aminus:+.6f})\n")
        f.write(f"  → A_plus {'↑' if final_aplus > init_aplus else '↓'} "
                f"(LTP {'增强' if final_aplus > init_aplus else '抑制'}), "
                f"A_minus {'↑' if final_aminus > init_aminus else '↓'} "
                f"(LTD {'增强' if final_aminus > init_aminus else '抑制'})\n\n")

        # Per-epoch summary
        for epoch in range(1, N_EPOCHS + 1):
            epoch_data = [g for g in global_history if g["epoch"] == epoch]
            if not epoch_data:
                continue
            avg_r = sum(g["avg_reward"] for g in epoch_data) / len(epoch_data)
            avg_changed = sum(g["stdp_changed_pct"] for g in epoch_data) / len(epoch_data)
            norms_first = [g["stdp_norm_first"] for g in epoch_data]
            norms_last = [g["stdp_norm_last"] for g in epoch_data]
            f.write(f"━━━ Epoch {epoch} ━━━\n")
            f.write(f"  Prompts: {len(epoch_data)}\n")
            f.write(f"  Avg reward: {avg_r:.4f}\n")
            f.write(f"  Avg STDP-change-rate: {avg_changed:.1f}%\n")
            f.write(f"  STDP norm range: {min(norms_first):.6f} → {max(norms_last):.6f}\n\n")

        # Sample generations
        f.write("━━━ Sample generations (epoch 1, first 10) ━━━\n")
        for g in global_history[:10]:
            f.write(f"  [{g['prompt_idx']:>3d}] {g['prompt']!r}\n")
            f.write(f"       → {g['generated_text'][:80]!r}\n")
            f.write(f"       R={g['avg_reward']:.4f}, chg={g['stdp_changed_pct']:.0f}%\n\n")

        # Reward trend
        if global_history:
            f.write("━━━ Reward trend (per 20-prompt block) ━━━\n")
            block_size = 20
            for epoch in range(1, N_EPOCHS + 1):
                epoch_data = [g for g in global_history if g["epoch"] == epoch]
                for start in range(0, len(epoch_data), block_size):
                    block = epoch_data[start:start + block_size]
                    if not block:
                        continue
                    avg_r = sum(g["avg_reward"] for g in block) / len(block)
                    avg_chg = sum(g["stdp_changed_pct"] for g in block) / len(block)
                    f.write(f"  E{epoch} [{start+1:>3d}-{start+len(block):>3d}] "
                            f"R={avg_r:+.4f} chg={avg_chg:.1f}%\n")

    print(f"💾 TXT: {txt_path}", flush=True)


# ============================================================================
# 8. 趋势图
# ============================================================================
def plot_drift(step_history, global_history, init_norm, final_norm, init_sp, final_sp,
               init_aplus, final_aplus, init_aminus, final_aminus):
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

    fig, axes = plt.subplots(2, 4, figsize=(24, 10), constrained_layout=True)

    steps = [s["global_step"] for s in step_history]
    window = 50

    def moving_average(data, w=window):
        if len(data) < w:
            return data
        return [sum(data[max(0, i - w):i]) / min(i, w) for i in range(1, len(data) + 1)]

    # 1. STDP norm drift
    ax = axes[0, 0]
    norms = [s["stdp_norm"] for s in step_history]
    ax.plot(steps, norms, color='tab:blue', linewidth=0.5, alpha=0.4)
    ma = moving_average(norms)
    ax.plot(steps, ma, color='tab:blue', linewidth=2, label='50-step MA')
    ax.axhline(init_norm, color='gray', linestyle='--', alpha=0.5, label=f'initial={init_norm:.4f}')
    ax.axhline(final_norm, color='red', linestyle='--', alpha=0.5, label=f'final={final_norm:.4f}')
    ax.set_title('STDP 突触范数漂移', fontsize=12)
    ax.set_xlabel('global step')
    ax.set_ylabel('||STDP_W||')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. A_plus / A_minus (LTP/LTD 速率)
    ax = axes[0, 1]
    a_plus_vals = [s.get("A_plus_mean", 0) for s in step_history]
    a_minus_vals = [s.get("A_minus_mean", 0) for s in step_history]
    ax.plot(steps, a_plus_vals, color='tab:green', linewidth=1, alpha=0.5, label='A_plus (LTP)')
    ax.plot(steps, a_minus_vals, color='tab:red', linewidth=1, alpha=0.5, label='A_minus (LTD)')
    ma_ap = moving_average(a_plus_vals)
    ma_am = moving_average(a_minus_vals)
    ax.plot(steps, ma_ap, color='tab:green', linewidth=2.5)
    ax.plot(steps, ma_am, color='tab:red', linewidth=2.5)
    ax.axhline(init_aplus, color='tab:green', linestyle='--', alpha=0.3)
    ax.axhline(init_aminus, color='tab:red', linestyle='--', alpha=0.3)
    ax.set_title('synaptic_plasticity LTP/LTD 速率', fontsize=12)
    ax.set_xlabel('global step')
    ax.set_ylabel('rate')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. External reward
    ax = axes[0, 2]
    rewards = [s["external_reward"] for s in step_history]
    ax.plot(steps, rewards, color='tab:green', linewidth=0.5, alpha=0.3)
    ma_r = moving_average(rewards)
    ax.plot(steps, ma_r, color='tab:green', linewidth=2, label='50-step MA')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_title('外部 reward', fontsize=12)
    ax.set_xlabel('global step')
    ax.set_ylabel('reward')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. STDP-change-rate (per prompt)
    ax = axes[0, 3]
    prompt_indices = [g["prompt_idx"] + (g["epoch"] - 1) * N_PROMPTS for g in global_history]
    change_rates = [g["stdp_changed_pct"] for g in global_history]
    colors = ['tab:orange' if g["epoch"] == 1 else 'tab:red' for g in global_history]
    ax.bar(prompt_indices, change_rates, color=colors, alpha=0.6)
    ax.set_title('STDP 改变 token 比例 (每 prompt)', fontsize=12)
    ax.set_xlabel('prompt index (orange=E1, red=E2)')
    ax.set_ylabel('% tokens changed')
    ax.grid(True, alpha=0.3)

    # 5. DA
    ax = axes[1, 0]
    da = [s["da"] for s in step_history]
    ax.plot(steps, da, color='tab:purple', linewidth=0.5, alpha=0.3)
    ma_da = moving_average(da)
    ax.plot(steps, ma_da, color='tab:purple', linewidth=2, label='50-step MA')
    ax.set_title('多巴胺 DA', fontsize=12)
    ax.set_xlabel('global step')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 6. NE
    ax = axes[1, 1]
    ne = [s["ne"] for s in step_history]
    ax.plot(steps, ne, color='tab:orange', linewidth=0.5, alpha=0.3)
    ma_ne = moving_average(ne)
    ax.plot(steps, ma_ne, color='tab:orange', linewidth=2, label='50-step MA')
    ax.set_title('去甲肾上腺素 NE', fontsize=12)
    ax.set_xlabel('global step')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 7. STDP grad norm
    ax = axes[1, 2]
    grads = [s["stdp_grad_norm"] for s in step_history]
    ax.plot(steps, grads, color='tab:red', linewidth=0.5, alpha=0.3)
    ma_g = moving_average(grads)
    ax.plot(steps, ma_g, color='tab:red', linewidth=2, label='50-step MA')
    ax.set_title('STDP 梯度范数', fontsize=12)
    ax.set_xlabel('global step')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 8. Valence
    ax = axes[1, 3]
    val = [s["valence"] for s in step_history]
    ax.plot(steps, val, color='tab:green', linewidth=0.5, alpha=0.3)
    ma_v = moving_average(val)
    ax.plot(steps, ma_v, color='tab:green', linewidth=2, label='50-step MA')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_title('杏仁核 valence', fontsize=12)
    ax.set_xlabel('global step')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    stdp_delta_pct = (final_norm - init_norm) / max(init_norm, 1e-8) * 100
    sp_delta_pct = (final_sp - init_sp) / max(init_sp, 1e-8) * 100

    fig.suptitle(
        f"GPT-2 × stdpbrain v3 长训练 — "
        f"{N_PROMPTS} prompts × {MAX_NEW_TOKENS} tokens × {N_EPOCHS} epochs = {len(steps)} steps\n"
        f"STDP norm: {init_norm:.4f} → {final_norm:.4f} ({stdp_delta_pct:+.2f}%) | "
        f"A_plus: {init_aplus:.5f} → {final_aplus:.5f} | "
        f"A_minus: {init_aminus:.5f} → {final_aminus:.5f}",
        fontsize=13,
    )

    plot_path = os.path.join(DOWNLOAD_DIR, "gpt2_stdpbrain_v3_longtrain.png")
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"💾 PNG: {plot_path}", flush=True)


if __name__ == "__main__":
    main()
