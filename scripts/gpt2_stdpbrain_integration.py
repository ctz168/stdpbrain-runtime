#!/usr/bin/env python3
"""
GPT-2 (from lal) × stdpbrain 类脑神经活动整合测试
==================================================

lal 仓库中的 GPT-2 模型即 HuggingFace 标准的 `gpt2`（12 层 / 768 embd /
12 head / 50257 vocab / 1024 ctx）—— 既是 lal/bench_pytorch.py 的基准模型，
也是 lal/tools/export_gpt2_full.py 导出到 C 推理服务器 (gpt2_server.c)
的同一组权重。

本脚本把该 GPT-2 作为"新皮层基座"接入 stdpbrain 的脑模块（前额叶 / 基底节-
多巴胺 / 蓝斑-NE / 杏仁核 / 小脑纠错 / 双系统 / 信号总线 / STDP 三因子），
让 GPT-2 在生成 token 的同时驱动一整套类脑神经活动，并实时观测：
- 多巴胺 DA（奖赏预测误差）
- 去甲肾上腺素 NE（唤醒度）
- 小脑预测误差
- 杏仁核情绪效价 valence
- 双系统 S1 快 / S2 慢
- STDP 突触权重范数演化

所有 brain 模块以 hidden_size=768 直接吃 GPT-2 最后一层隐状态——不做投影，
保留 GPT-2 表征的全部信息。
"""

from __future__ import annotations

import os
import sys
import json
import time
import warnings
from typing import Any, Dict, List, Tuple

warnings.filterwarnings("ignore")

# === 路径 setup ===
STDPBRAIN_DIR = "/home/z/my-project/repos/stdpbrain"
LAL_DIR = "/home/z/my-project/repos/lal"
DOWNLOAD_DIR = "/home/z/my-project/download"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.chdir(STDPBRAIN_DIR)
sys.path.insert(0, STDPBRAIN_DIR)

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn.functional as F

# === 全局常量 ===
HIDDEN_SIZE = 768           # GPT-2 n_embd，brain 模块直接对接
BATCH = 1                   # 单序列生成
DEVICE = "cpu"
MAX_NEW_TOKENS = 25         # 每个提示词生成 25 个 token
TORCH_THREADS = 4
torch.set_num_threads(TORCH_THREADS)


# ============================================================================
# 1. 加载 GPT-2（lal 仓库的 gpt2 模型）
# ============================================================================
def load_gpt2():
    """加载 lal/bench_pytorch.py 中使用的同款 GPT-2 模型。"""
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    print("[1/4] 加载 GPT-2 (lal 仓库参考模型)...", flush=True)
    t0 = time.time()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    model.to(DEVICE)
    # 关闭 gradient
    for p in model.parameters():
        p.requires_grad_(False)
    dt = time.time() - t0
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      ✅ GPT-2 加载完成 | 参数 {n_params/1e6:.1f}M | 耗时 {dt:.1f}s", flush=True)
    print(f"      config: n_layer={model.config.n_layer} n_embd={model.config.n_embd} "
          f"n_head={model.config.n_head} vocab={model.config.vocab_size}", flush=True)
    return tokenizer, model


# ============================================================================
# 2. 加载 stdpbrain 脑模块（hidden_size=768 对齐 GPT-2）
# ============================================================================
def load_brain_modules():
    """加载 stdpbrain 的核心脑模块，全部用 hidden_size=768 对齐 GPT-2。"""
    print("[2/4] 加载 stdpbrain 脑模块 (hidden_size=768)...", flush=True)
    import importlib

    modules: Dict[str, Any] = {}
    spec = [
        ("cerebellar",       "core.cerebellar_correction_677",    "create_cerebellar_correction_system",  dict(hidden_size=HIDDEN_SIZE)),
        ("basal_ganglia",    "core.basal_ganglia_dopamine",       "create_basal_ganglia_dopamine_system", dict(hidden_size=HIDDEN_SIZE)),
        ("lc_ne",            "core.locus_coeruleus_ne",           "create_locus_coeruleus_ne_system",     dict(hidden_size=HIDDEN_SIZE)),
        ("astrocyte",        "core.astrocyte_neuromod_coupling",  "create_astrocyte_neuromod_coupling",   dict(hidden_size=HIDDEN_SIZE)),
        ("amygdala",         "core.amygdala",                     "create_amygdala_system",               dict(hidden_size=HIDDEN_SIZE)),
        ("dual_process",     "core.dual_process",                 "create_dual_process_system",           dict(hidden_size=HIDDEN_SIZE)),
        ("synaptic_plast",   "core.synaptic_plasticity",          "create_synaptic_plasticity_system",    dict(hidden_size=HIDDEN_SIZE)),
        ("signal_bus",       "core.signal_bus",                   "create_signal_bus",                    dict()),
        ("adv_neuromod",     "core.advanced_neuromodulation",     "create_advanced_neuromodulation_system", dict(hidden_size=HIDDEN_SIZE)),
    ]

    total_params = 0
    for name, path, factory, kwargs in spec:
        t0 = time.time()
        try:
            mod = importlib.import_module(path)
            f = getattr(mod, factory)
            m = f(**kwargs)
            m.eval() if hasattr(m, "eval") else None
            for p in m.parameters():
                p.requires_grad_(False)
            n = sum(p.numel() for p in m.parameters()) if hasattr(m, "parameters") else 0
            total_params += n
            modules[name] = m
            print(f"      ✅ {name:<16} | {n:>10,} params | {(time.time()-t0)*1000:>6.1f}ms", flush=True)
        except Exception as e:
            print(f"      ❌ {name:<16} | ERROR: {e}", flush=True)
    print(f"      合计 brain 参数: {total_params:,}", flush=True)
    return modules


# ============================================================================
# 3. 一步"脑活动"计算：GPT-2 隐状态 → 脑模块 → 神经活动指标
# ============================================================================
@torch.no_grad()
def brain_step(
    modules: Dict[str, Any],
    hidden_now: torch.Tensor,        # (1, 768) — GPT-2 当前 token 的最后一层隐状态
    hidden_prev: torch.Tensor,       # (1, 768) — 上一个 token 的隐状态（用于"预测"）
    hidden_next: torch.Tensor,       # (1, 768) — 下一个 token 的隐状态（target）
    syn_weights: torch.Tensor,       # (768, 768) — STDP 维护的突触矩阵
    step_idx: int,
) -> Tuple[Dict[str, Any], torch.Tensor]:
    """跑一遍脑模块闭环，返回该 token 的脑活动快照 + 更新后的突触权重。"""
    snap: Dict[str, Any] = {"step": step_idx}

    # ---- P6 小脑纠错：比较"预测"(prev→now 一步预测) 与"实际"(now) ----
    # actual_output, predicted_output, target
    cb = modules.get("cerebellar")
    if cb is not None:
        try:
            r = cb.forward(hidden_now, hidden_prev, hidden_next)
            err = r.get("prediction_error", None)
            if isinstance(err, torch.Tensor):
                err_val = float(err.norm().item())
            else:
                err_val = float(err) if err is not None else 0.0
            snap["cerebellar_error"] = err_val
            corrected = r.get("corrected_output", hidden_now)
            if not isinstance(corrected, torch.Tensor):
                corrected = hidden_now
        except Exception as e:
            snap["cerebellar_error"] = 0.0
            corrected = hidden_now
    else:
        snap["cerebellar_error"] = 0.0
        corrected = hidden_now

    # ---- P7 基底节-多巴胺：误差驱动 DA 释放 ----
    bg = modules.get("basal_ganglia")
    da_level = 0.5
    if bg is not None:
        try:
            r = bg.forward(corrected)
            if hasattr(bg, "get_dopamine_level"):
                da = bg.get_dopamine_level()
                if isinstance(da, torch.Tensor):
                    da_level = float(da.mean().item())
                else:
                    da_level = float(da)
            bg_out = r.get("output", corrected)
            if not isinstance(bg_out, torch.Tensor):
                bg_out = corrected
        except Exception as e:
            bg_out = corrected
    else:
        bg_out = corrected
    snap["da"] = da_level

    # ---- P8 蓝斑-NE：DA 驱动 NE 释放 ----
    lc = modules.get("lc_ne")
    ne_level = 0.5
    if lc is not None:
        try:
            r = lc.forward(bg_out)
            if hasattr(lc, "get_ne_level"):
                ne = lc.get_ne_level()
                if isinstance(ne, torch.Tensor):
                    ne_level = float(ne.mean().item())
                else:
                    ne_level = float(ne)
            lc_out = r.get("output", bg_out)
            if not isinstance(lc_out, torch.Tensor):
                lc_out = bg_out
        except Exception:
            lc_out = bg_out
    else:
        lc_out = bg_out
    snap["ne"] = ne_level

    # ---- P9 胶质耦合：DA+NE 驱动清除 ----
    ast = modules.get("astrocyte")
    clearance = 0.0
    if ast is not None:
        try:
            ast.forward(da_level=da_level, ne_level=ne_level, ach_level=0.5, sht_level=0.5)
            if hasattr(ast, "get_glymphatic_rate"):
                clearance = float(ast.get_glymphatic_rate())
        except Exception:
            pass
    snap["clearance"] = clearance

    # ---- 杏仁核：情绪效价 ----
    amy = modules.get("amygdala")
    valence = 0.0
    if amy is not None:
        try:
            amy_in = lc_out
            if amy_in.dim() == 1:
                amy_in = amy_in.unsqueeze(0)
            r = amy.forward(amy_in)
            stats = r.get("stats", {})
            v = stats.get("valence", 0)
            if isinstance(v, torch.Tensor):
                v = float(v.mean().item())
            valence = float(v)
            amy_out = r.get("output", amy_in)
            if not isinstance(amy_out, torch.Tensor):
                amy_out = amy_in
        except Exception:
            amy_out = lc_out
    else:
        amy_out = lc_out
    snap["valence"] = valence

    # ---- 双系统：S1 快 / S2 慢 ----
    dp = modules.get("dual_process")
    system_mode = "?"
    if dp is not None:
        try:
            r = dp.forward(amy_out)
            s = r.get("stats", {}).get("active_system", "?")
            system_mode = str(s)
        except Exception:
            pass
    snap["system"] = system_mode

    # ---- STDP 三因子：更新突触权重 ----
    sp = modules.get("synaptic_plast")
    if sp is not None:
        try:
            new_w = sp.forward(
                pre_activity=hidden_prev,
                post_activity=hidden_now,
                dopamine_level=da_level,
                weights=syn_weights,
            )
            if isinstance(new_w, torch.Tensor):
                syn_weights = new_w
        except Exception:
            pass
    snap["syn_norm"] = float(syn_weights.norm().item())

    # ---- 信号总线：广播当前快照 ----
    sb = modules.get("signal_bus")
    if sb is not None:
        try:
            sb.forward()
        except Exception:
            pass

    return snap, syn_weights


# ============================================================================
# 4. 跑一个 prompt：GPT-2 生成 + 脑活动记录
# ============================================================================
@torch.no_grad()
def run_one_prompt(
    tokenizer, model, modules: Dict[str, Any],
    prompt: str, max_new_tokens: int = MAX_NEW_TOKENS,
) -> Dict[str, Any]:
    """对单个 prompt 用 GPT-2 贪心生成，逐 token 记录脑活动。"""
    print(f"\n{'─' * 78}", flush=True)
    print(f"📌 Prompt: {prompt!r}", flush=True)
    print(f"{'─' * 78}", flush=True)

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)  # (1, L)
    generated_ids = input_ids.clone()
    L0 = input_ids.shape[1]

    # 初始化 STDP 突触矩阵
    syn_weights = torch.randn(HIDDEN_SIZE, HIDDEN_SIZE).clamp(-1, 1) * 0.05

    # 先跑一次 forward 拿到 prompt 最后一个 token 的隐状态作为 hidden_prev
    out = model(input_ids, output_hidden_states=True, use_cache=False)
    hidden_states_all = out.hidden_states  # tuple of (1, L, 768), len=n_layer+1
    hidden_prev = hidden_states_all[-1][:, -1, :].squeeze(0).detach().clone()  # (768,)
    if hidden_prev.dim() == 1:
        hidden_prev = hidden_prev.unsqueeze(0)  # (1, 768)

    snapshots: List[Dict[str, Any]] = []
    generated_text_tokens: List[str] = []

    print(f"  step | token            | DA     | NE     | err    | valence | sys | syn_norm", flush=True)
    print(f"  -----+------------------+--------+--------+--------+---------+-----+---------", flush=True)

    t_gen_start = time.time()
    for step in range(max_new_tokens):
        # 贪心解码：取 argmax
        logits = out.logits[:, -1, :]  # (1, vocab)
        next_id = int(torch.argmax(logits, dim=-1).item())
        generated_ids = torch.cat([generated_ids, torch.tensor([[next_id]], device=DEVICE)], dim=1)
        token_str = tokenizer.decode([next_id])
        generated_text_tokens.append(token_str)

        # 再跑一次 forward 拿新的 hidden state（无 KV cache，简洁优先）
        out = model(generated_ids, output_hidden_states=True, use_cache=False)
        hidden_now = out.hidden_states[-1][:, -1, :].squeeze(0).detach().clone()
        if hidden_now.dim() == 1:
            hidden_now = hidden_now.unsqueeze(0)

        # 预测"下一步"的 target —— 用当前 token 的 hidden state 作为 next target 的代理
        # （因为我们还没生成下一个 token；这里 hidden_next 是当前 token 的 hidden
        #   state，代表"实际发生的"；hidden_prev 是上一步"预测的"）
        hidden_next = hidden_now.clone()

        # 跑脑模块
        snap, syn_weights = brain_step(
            modules, hidden_now, hidden_prev, hidden_next, syn_weights, step
        )
        snap["token"] = token_str
        snap["token_id"] = next_id
        snapshots.append(snap)

        # 打印本步
        tok_disp = repr(token_str)[:16]
        print(f"  {step:>4d} | {tok_disp:<16} | {snap['da']:.4f} | {snap['ne']:.4f} | "
              f"{snap['cerebellar_error']:>6.3f} | {snap['valence']:+.3f}   | "
              f"{snap['system']:<3s} | {snap['syn_norm']:.3f}", flush=True)

        # 滚动
        hidden_prev = hidden_now.clone()

        # 遇到 EOS 终止
        if next_id == tokenizer.eos_token_id:
            print(f"  (EOS, stop)", flush=True)
            break

    dt = time.time() - t_gen_start
    full_text = prompt + "".join(generated_text_tokens)
    print(f"\n  ⏱ 生成 {len(snapshots)} tokens | 耗时 {dt:.2f}s | "
          f"{len(snapshots)/max(dt,1e-6):.2f} tok/s", flush=True)
    print(f"  📝 完整文本: {full_text!r}", flush=True)

    return {
        "prompt": prompt,
        "generated_text": full_text,
        "new_tokens": generated_text_tokens,
        "snapshots": snapshots,
        "gen_time_s": dt,
        "tokens_per_sec": len(snapshots) / max(dt, 1e-6),
    }


# ============================================================================
# 5. 主流程
# ============================================================================
def main():
    print("=" * 78, flush=True)
    print("🧠 GPT-2 (from lal) × stdpbrain 类脑神经活动整合测试", flush=True)
    print("=" * 78, flush=True)
    print(f"环境: torch={torch.__version__} device={DEVICE} threads={TORCH_THREADS}", flush=True)
    print(f"hidden_size={HIDDEN_SIZE} (= GPT-2 n_embd, 直接对接，无投影)", flush=True)

    tokenizer, gpt2 = load_gpt2()
    modules = load_brain_modules()

    # 测试 prompt — 覆盖不同语义场景
    prompts = [
        "The capital of France is",                # lal/bench_pytorch.py 同款
        "Once upon a time",                        # lal/bench_pytorch.py 同款
        "Hello, how are",                          # lal/bench_pytorch.py 同款
        "Machine learning is",                     # lal/bench_pytorch.py 同款
        "The meaning of life is",                  # 哲学性
    ]

    print(f"\n[3/4] 跑 {len(prompts)} 个 prompt 的 GPT-2 生成 + 脑活动记录...", flush=True)
    all_results: List[Dict[str, Any]] = []
    for p in prompts:
        try:
            r = run_one_prompt(tokenizer, gpt2, modules, p)
            all_results.append(r)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  ❌ prompt {p!r} 失败: {e}", flush=True)

    # === 汇总 ===
    print(f"\n[4/4] 汇总脑活动 & 写报告...", flush=True)
    summary = summarize(all_results)
    print(summary["text"], flush=True)

    # 保存 JSON
    json_path = os.path.join(DOWNLOAD_DIR, "gpt2_stdpbrain_activity.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "model": "gpt2 (lal repo reference)",
                "hidden_size": HIDDEN_SIZE,
                "max_new_tokens": MAX_NEW_TOKENS,
                "device": DEVICE,
                "torch_threads": TORCH_THREADS,
                "n_prompts": len(prompts),
            },
            "results": all_results,
            "summary": summary["dict"],
        }, f, ensure_ascii=False, indent=2)
    print(f"\n💾 JSON 报告: {json_path}", flush=True)

    # 保存文本报告
    txt_path = os.path.join(DOWNLOAD_DIR, "gpt2_stdpbrain_activity.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 78 + "\n")
        f.write("GPT-2 (from lal) × stdpbrain 类脑神经活动整合测试报告\n")
        f.write("=" * 78 + "\n\n")
        f.write(f"模型: gpt2 (lal 仓库的参考实现, 12 层 / 768 embd / 12 head / 50257 vocab)\n")
        f.write(f"脑模块 hidden_size: {HIDDEN_SIZE} (= GPT-2 n_embd, 无投影直接对接)\n")
        f.write(f"设备: {DEVICE}, torch {torch.__version__}, threads={TORCH_THREADS}\n")
        f.write(f"每 prompt 最大新 token: {MAX_NEW_TOKENS}\n\n")
        for r in all_results:
            f.write("─" * 78 + "\n")
            f.write(f"Prompt: {r['prompt']!r}\n")
            f.write(f"生成: {r['generated_text']!r}\n")
            f.write(f"耗时: {r['gen_time_s']:.2f}s | {r['tokens_per_sec']:.2f} tok/s\n")
            f.write("  step | token            | DA     | NE     | err    | valence | sys | syn_norm\n")
            f.write("  -----+------------------+--------+--------+--------+---------+-----+---------\n")
            for s in r["snapshots"]:
                tok_disp = repr(s["token"])[:16]
                f.write(f"  {s['step']:>4d} | {tok_disp:<16} | {s['da']:.4f} | {s['ne']:.4f} | "
                        f"{s['cerebellar_error']:>6.3f} | {s['valence']:+.3f}   | "
                        f"{s['system']:<3s} | {s['syn_norm']:.3f}\n")
            f.write("\n")
        f.write("\n" + "=" * 78 + "\n")
        f.write("汇总\n")
        f.write("=" * 78 + "\n")
        f.write(summary["text"])
    print(f"💾 文本报告: {txt_path}", flush=True)

    # 保存脑活动趋势图
    try:
        plot_path = os.path.join(DOWNLOAD_DIR, "gpt2_stdpbrain_activity.png")
        plot_activity(all_results, plot_path)
        print(f"💾 脑活动趋势图: {plot_path}", flush=True)
    except Exception as e:
        print(f"  ⚠️ 趋势图生成失败: {e}", flush=True)

    print("\n✅ 测试完成", flush=True)


# ============================================================================
# 汇总
# ============================================================================
def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {"text": "(无结果)", "dict": {}}

    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("🧠 脑活动汇总")
    lines.append("=" * 78)

    # 全局聚合
    all_da, all_ne, all_err, all_val, all_syn = [], [], [], [], []
    per_prompt_summary = []
    for r in results:
        snaps = r["snapshots"]
        if not snaps:
            continue
        da = [s["da"] for s in snaps]
        ne = [s["ne"] for s in snaps]
        err = [s["cerebellar_error"] for s in snaps]
        val = [s["valence"] for s in snaps]
        syn = [s["syn_norm"] for s in snaps]
        all_da += da; all_ne += ne; all_err += err; all_val += val; all_syn += syn

        # 系统1/2分布
        s1_count = sum(1 for s in snaps if "1" in s.get("system", ""))
        s2_count = sum(1 for s in snaps if "2" in s.get("system", ""))

        ps = {
            "prompt": r["prompt"],
            "n_tokens": len(snaps),
            "tok_per_sec": r["tokens_per_sec"],
            "da_mean": sum(da)/len(da),
            "ne_mean": sum(ne)/len(ne),
            "err_mean": sum(err)/len(err),
            "valence_mean": sum(val)/len(val),
            "syn_norm_first": syn[0] if syn else 0,
            "syn_norm_last": syn[-1] if syn else 0,
            "syn_delta_pct": ((syn[-1]-syn[0])/max(abs(syn[0]),1e-8)*100) if syn else 0,
            "s1_count": s1_count,
            "s2_count": s2_count,
            "generated": r["generated_text"],
        }
        per_prompt_summary.append(ps)

        lines.append(f"\n📌 {r['prompt']!r}")
        lines.append(f"   生成 {ps['n_tokens']} tokens | {ps['tok_per_sec']:.2f} tok/s")
        lines.append(f"   DA 均值={ps['da_mean']:.4f}  NE 均值={ps['ne_mean']:.4f}  "
                     f"误差均值={ps['err_mean']:.4f}  valence 均值={ps['valence_mean']:+.4f}")
        lines.append(f"   突触范数: {ps['syn_norm_first']:.3f} → {ps['syn_norm_last']:.3f} "
                     f"({ps['syn_delta_pct']:+.1f}%)")
        lines.append(f"   双系统: S1快={s1_count}  S2慢={s2_count}")
        lines.append(f"   生成文本: {r['generated_text']!r}")

    if all_da:
        lines.append("\n" + "─" * 78)
        lines.append("📊 全局统计 (所有 prompt 所有 token 聚合):")
        lines.append(f"   总 token 数: {len(all_da)}")
        lines.append(f"   DA        : mean={sum(all_da)/len(all_da):.4f}  "
                     f"min={min(all_da):.4f}  max={max(all_da):.4f}")
        lines.append(f"   NE        : mean={sum(all_ne)/len(all_ne):.4f}  "
                     f"min={min(all_ne):.4f}  max={max(all_ne):.4f}")
        lines.append(f"   小脑误差  : mean={sum(all_err)/len(all_err):.4f}  "
                     f"min={min(all_err):.4f}  max={max(all_err):.4f}")
        lines.append(f"   valence   : mean={sum(all_val)/len(all_val):+.4f}  "
                     f"min={min(all_val):+.4f}  max={max(all_val):+.4f}")
        lines.append(f"   突触范数  : first={all_syn[0]:.3f}  last={all_syn[-1]:.3f}")

        # 解读
        lines.append("\n" + "─" * 78)
        lines.append("🔍 神经科学解读:")
        mean_da = sum(all_da)/len(all_da)
        mean_ne = sum(all_ne)/len(all_ne)
        mean_val = sum(all_val)/len(all_val)
        syn_delta = ((all_syn[-1]-all_syn[0])/max(abs(all_syn[0]),1e-8)*100) if all_syn else 0

        if mean_da > 0.55:
            lines.append(f"   • DA={mean_da:.3f} 偏高 → 奖赏信号活跃，GPT-2 隐状态在基底节看来是"
                         "「正向预期」的（高熵→低熵的预测收敛）")
        elif mean_da < 0.45:
            lines.append(f"   • DA={mean_da:.3f} 偏低 → 奖赏预测误差为负，模型可能处于「不确定」"
                         "或低奖赏的语义区间")
        else:
            lines.append(f"   • DA={mean_da:.3f} 中性 → 基底节处于稳态")

        if mean_ne > 0.5:
            lines.append(f"   • NE={mean_ne:.3f} 偏高 → 蓝斑唤醒度高，注意力聚焦（通常出现在"
                     "信息密集或情感强烈的 prompt）")
        else:
            lines.append(f"   • NE={mean_ne:.3f} 偏低 → 唤醒度低，平稳生成")

        if mean_val > 0.05:
            lines.append(f"   • valence={mean_val:+.3f} 正向偏移 → 杏仁核判定 GPT-2 输出整体情绪偏积极")
        elif mean_val < -0.05:
            lines.append(f"   • valence={mean_val:+.3f} 负向偏移 → 杏仁核判定 GPT-2 输出整体情绪偏消极")
        else:
            lines.append(f"   • valence={mean_val:+.3f} 中性 → 情绪效价平衡")

        if syn_delta > 5:
            lines.append(f"   • STDP 突触范数 +{syn_delta:.1f}% → LTP（长时程增强）主导，"
                     "突触在被强化，相当于「在学习」")
        elif syn_delta < -5:
            lines.append(f"   • STDP 突触范数 {syn_delta:.1f}% → LTD（长时程抑制）主导，"
                     "突触在被弱化")
        else:
            lines.append(f"   • STDP 突触范数变化 {syn_delta:+.1f}% → 稳态，LTP/LTD 平衡")

    text = "\n".join(lines)
    return {"text": text, "dict": {"per_prompt": per_prompt_summary}}


# ============================================================================
# 趋势图
# ============================================================================
def plot_activity(results: List[Dict[str, Any]], out_path: str):
    """画 4 个 prompt × 4 个指标 (DA/NE/err/valence) 的脑活动趋势图。"""
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

    # 最多画 4 个 prompt
    results = results[:4]
    if not results:
        return

    fig, axes = plt.subplots(len(results), 4, figsize=(18, 4 * len(results)),
                             constrained_layout=True, sharex=False)
    if len(results) == 1:
        axes = axes.reshape(1, -1)

    metrics = [
        ("DA 多巴胺", "da", "tab:purple"),
        ("NE 去甲肾上腺素", "ne", "tab:orange"),
        ("小脑预测误差", "cerebellar_error", "tab:red"),
        ("杏仁核 valence", "valence", "tab:green"),
    ]

    for i, r in enumerate(results):
        snaps = r["snapshots"]
        if not snaps:
            continue
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

    fig.suptitle(
        "GPT-2 (from lal) × stdpbrain 类脑神经活动 — 逐 token 趋势\n"
        f"hidden_size=768, 每行 1 个 prompt, 每列 1 个指标",
        fontsize=13,
    )
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
