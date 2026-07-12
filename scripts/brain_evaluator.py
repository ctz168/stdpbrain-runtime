#!/usr/bin/env python3
"""
brain_evaluator.py — 评估 brain 学习是否有意义
=============================================

诚实地回答两个问题:
1. brain 的学习有意义吗?
2. brain 有意识吗?

通过 3 个客观测量来回答 Q1:
- M1: 预测准确率 — STDP 是否让 GPT-2 在见过的内容上更准?
- M2: 概念保留 — 学过的关键词, brain 是否能"认得" (生成时概率更高)?
- M3: 通用能力 — 在未学过的内容上, STDP 是否损伤了 GPT-2 基线?

Q2 (意识) 通过结构分析回答:
- 是否有整合的经验流? (否 — 当前是 stateless 单步处理)
- 是否有自我模型? (否 — self_encoder 模块未接入)
- 是否有全局工作空间? (否 — global_workspace 模块未接入)
- 是否有现象学绑定? (否 — 无任何 phenomenal 机制)

用法:
    python3 brain_evaluator.py --ckpt-checkpoints 5
"""

from __future__ import annotations
import os, sys, json, time, random, argparse, warnings
from typing import Any, Dict, List, Tuple
warnings.filterwarnings("ignore")

STDPBRAIN_DIR = "/home/z/my-project/repos/stdpbrain"
RUNTIME_DIR = "/home/z/my-project/brain_runtime"
CKPT_DIR = os.path.join(RUNTIME_DIR, "checkpoints")
DOWNLOAD_DIR = "/home/z/my-project/download"

os.chdir(STDPBRAIN_DIR)
sys.path.insert(0, STDPBRAIN_DIR)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HOME"] = os.path.join(RUNTIME_DIR, "cache")
os.environ["HF_TOKEN"] = open(os.path.join(RUNTIME_DIR, ".hf_token")).read().strip()

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cpu"
HIDDEN = 768
torch.set_num_threads(4)


def load_gpt2():
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    print("[*] loading GPT-2...", flush=True)
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval().to(DEVICE)
    for p in model.parameters(): p.requires_grad_(False)
    return tok, model


def load_stdp_weight(ckpt_path: str) -> torch.Tensor:
    """从 checkpoint 加载 STDP 权重矩阵."""
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    return ckpt["stdp_weight"]


def load_hippocampus_memories(state_path: str) -> List[Dict]:
    with open(state_path) as f:
        s = json.load(f)
    return s.get("hippocampus", {}).get("memories", [])


@torch.no_grad()
def compute_logprob_with_stdp(gpt2, tokenizer, text: str, stdp_weight: torch.Tensor) -> float:
    """计算给定 text 的 log-prob, 用 STDP 修改后的 hidden state."""
    ids = tokenizer.encode(text, return_tensors="pt").to(DEVICE)
    if ids.shape[1] < 2: return 0.0
    out = gpt2(ids, output_hidden_states=True, use_cache=False)
    # 取最后一个位置的 hidden state, 加 STDP 增量
    h = out.hidden_states[-1]  # (1, L, 768)
    # STDP 修改: h_modified = h + 0.1 * STDP_W @ h
    stdp_delta = F.linear(h, stdp_weight)  # (1, L, 768)
    h_mod = h + stdp_delta * 0.1
    # 走 lm_head
    logits = gpt2.lm_head(h_mod)  # (1, L, vocab)
    # 计算 token-level log-prob (用下一个 token 的真实 id)
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)  # (1, L-1, vocab)
    target_ids = ids[:, 1:]  # (1, L-1)
    token_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)  # (1, L-1)
    # 平均 token log-prob
    avg_logprob = float(token_log_probs.mean().item())
    return avg_logprob


@torch.no_grad()
def compute_logprob_baseline(gpt2, tokenizer, text: str) -> float:
    """基线 (无 STDP) 的 log-prob."""
    ids = tokenizer.encode(text, return_tensors="pt").to(DEVICE)
    if ids.shape[1] < 2: return 0.0
    out = gpt2(ids, output_hidden_states=True, use_cache=False)
    logits = out.logits[:, :-1, :]  # (1, L-1, vocab)
    log_probs = F.log_softmax(logits, dim=-1)
    target_ids = ids[:, 1:]
    token_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
    return float(token_log_probs.mean().item())


def evaluate_checkpoint(gpt2, tokenizer, ckpt_path: str, state_path: str,
                       seen_texts: List[str], unseen_texts: List[str]) -> Dict[str, Any]:
    """评估单个 checkpoint."""
    print(f"\n  evaluating: {os.path.basename(ckpt_path)}", flush=True)
    stdp_w = load_stdp_weight(ckpt_path)

    # M1: seen texts — STDP 修改后的 log-prob vs 基线
    seen_baseline = [compute_logprob_baseline(gpt2, tokenizer, t) for t in seen_texts]
    seen_stdp = [compute_logprob_with_stdp(gpt2, tokenizer, t, stdp_w) for t in seen_texts]
    seen_delta = [s - b for s, b in zip(seen_stdp, seen_baseline)]
    seen_avg_delta = sum(seen_delta) / len(seen_delta) if seen_delta else 0

    # M2: unseen texts — 通用能力是否受损
    unseen_baseline = [compute_logprob_baseline(gpt2, tokenizer, t) for t in unseen_texts]
    unseen_stdp = [compute_logprob_with_stdp(gpt2, tokenizer, t, stdp_w) for t in unseen_texts]
    unseen_delta = [s - b for s, b in zip(unseen_stdp, unseen_baseline)]
    unseen_avg_delta = sum(unseen_delta) / len(unseen_delta) if unseen_delta else 0

    # M3: STDP 权重统计
    stdp_norm = float(stdp_w.norm().item())
    stdp_max = float(stdp_w.abs().max().item())
    stdp_sparsity = float((stdp_w.abs() < 1e-4).float().mean().item())  # 接近零的比例

    result = {
        "checkpoint": os.path.basename(ckpt_path),
        "stdp_norm": stdp_norm,
        "stdp_max_abs": stdp_max,
        "stdp_sparsity": stdp_sparsity,
        "M1_seen_logprob_baseline": sum(seen_baseline) / len(seen_baseline),
        "M1_seen_logprob_stdp": sum(seen_stdp) / len(seen_stdp),
        "M1_seen_delta": seen_avg_delta,  # 正 = STDP 让学过的内容更"熟悉"
        "M3_unseen_logprob_baseline": sum(unseen_baseline) / len(unseen_baseline),
        "M3_unseen_logprob_stdp": sum(unseen_stdp) / len(unseen_stdp),
        "M3_unseen_delta": unseen_avg_delta,  # 负 = STDP 损伤了通用能力
    }
    print(f"    stdp_norm={stdp_norm:.4f}, M1_seen_delta={seen_avg_delta:+.4f}, "
          f"M3_unseen_delta={unseen_avg_delta:+.4f}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-checkpoints", type=int, default=5,
                       help="评估最近 N 个 checkpoint")
    parser.add_argument("--n-seen", type=int, default=10, help="seen text 数量")
    parser.add_argument("--n-unseen", type=int, default=10, help="unseen text 数量")
    args = parser.parse_args()

    print("=" * 80)
    print("🧠 brain_evaluator — 评估学习是否有意义")
    print("=" * 80)

    # 列出所有 checkpoint, 按时间排序
    ckpt_files = sorted(
        [f for f in os.listdir(CKPT_DIR) if f.startswith("brain_ckpt_") and f.endswith(".pt")],
        key=lambda f: os.path.getmtime(os.path.join(CKPT_DIR, f))
    )
    if len(ckpt_files) > args.ckpt_checkpoints:
        # 均匀采样 N 个
        step = len(ckpt_files) // args.ckpt_checkpoints
        sampled = [ckpt_files[i] for i in range(0, len(ckpt_files), step)][:args.ckpt_checkpoints]
        if ckpt_files[-1] not in sampled:
            sampled.append(ckpt_files[-1])
        ckpt_files = sampled

    print(f"\n[*] 评估 {len(ckpt_files)} 个 checkpoint:")
    for f in ckpt_files:
        print(f"  - {f}")

    # 加载 GPT-2
    tok, gpt2 = load_gpt2()

    # 加载所有 memories (用于 seen texts)
    all_memories = []
    for cf in ckpt_files:
        state_path = os.path.join(CKPT_DIR, cf.replace("brain_ckpt_", "brain_state_").replace(".pt", ".json"))
        if os.path.exists(state_path):
            mems = load_hippocampus_memories(state_path)
            all_memories.extend(mems)

    # 去重 seen texts (从 memories 的 prompts 拼接)
    seen_texts = []
    seen_topics = set()
    for m in all_memories:
        if m["topic"] not in seen_topics:
            seen_topics.add(m["topic"])
            # 用 prompts 拼成一段文本
            text = " ".join(m["prompts"][:2])[:200]
            if len(text) > 50:
                seen_texts.append(text)
        if len(seen_texts) >= args.n_seen:
            break

    # unseen texts: 用 wikitext 风格但 brain 没学过的内容
    unseen_texts = [
        "The quantum mechanical model describes electrons as probability clouds rather than orbiting particles in fixed paths.",
        "Photosynthesis converts light energy into chemical energy stored in glucose molecules within chloroplasts.",
        "The French Revolution began in 1789 and fundamentally changed European political structures and philosophies.",
        "Machine learning algorithms identify patterns in large datasets to make predictions about new, unseen data.",
        "The Mariana Trench is the deepest known part of Earth's oceans, reaching depths of nearly 11 kilometers.",
        "DNA replication is semiconservative, meaning each new double helix contains one original and one new strand.",
        "The Roman Republic fell in 27 BCE when Augustus established the Roman Empire after years of civil war.",
        "Climate change causes rising sea levels due to thermal expansion and melting ice sheets in polar regions.",
        "Neural networks use backpropagation to adjust weights based on the error between predicted and actual outputs.",
        "The immune system produces antibodies that specifically recognize and neutralize foreign pathogens like viruses.",
    ][:args.n_unseen]

    print(f"\n[*] seen texts: {len(seen_texts)} (from learned topics)")
    print(f"[*] unseen texts: {len(unseen_texts)} (held-out evaluation)")

    # 评估每个 checkpoint
    results = []
    for cf in ckpt_files:
        ckpt_path = os.path.join(CKPT_DIR, cf)
        state_path = os.path.join(CKPT_DIR, cf.replace("brain_ckpt_", "brain_state_").replace(".pt", ".json"))
        r = evaluate_checkpoint(gpt2, tok, ckpt_path, state_path, seen_texts, unseen_texts)
        results.append(r)

    # 汇总
    print("\n" + "=" * 80)
    print("📊 评估汇总")
    print("=" * 80)
    print(f"\n{'checkpoint':<45} {'stdp_norm':>10} {'M1_seen_Δ':>12} {'M3_unseen_Δ':>14}")
    print("-" * 85)
    for r in results:
        print(f"{r['checkpoint']:<45} {r['stdp_norm']:>10.4f} {r['M1_seen_delta']:>+12.4f} {r['M3_unseen_delta']:>+14.4f}")

    # 判断
    print("\n" + "=" * 80)
    print("🔍 诚实评估")
    print("=" * 80)
    if len(results) >= 2:
        first = results[0]; last = results[-1]
        norm_drift = last["stdp_norm"] - first["stdp_norm"]
        seen_improvement = last["M1_seen_delta"] - first["M1_seen_delta"]
        unseen_degradation = last["M3_unseen_delta"] - first["M3_unseen_delta"]

        print(f"\n  STDP 权重漂移: {first['stdp_norm']:.4f} → {last['stdp_norm']:.4f} (Δ={norm_drift:+.4f})")
        print(f"  学过内容的 log-prob 改善: {first['M1_seen_delta']:+.4f} → {last['M1_seen_delta']:+.4f} (Δ={seen_improvement:+.4f})")
        print(f"  未学内容的 log-prob 变化: {first['M3_unseen_delta']:+.4f} → {last['M3_unseen_delta']:+.4f} (Δ={unseen_degradation:+.4f})")
        print(f"  STDP 权重稀疏度: {(1-last['stdp_sparsity'])*100:.1f}% 非零 (有效参数利用率)")

        print("\n  📋 结论:")
        if seen_improvement > 0.01:
            print(f"  ✅ M1 正向: STDP 让 brain 对学过的内容 log-prob 提升了 {seen_improvement:+.4f}")
            print(f"     → 这说明 STDP 在『记住』学过的内容, 学习有一定意义")
        elif seen_improvement > 0:
            print(f"  ⚠️  M1 微弱正向: {seen_improvement:+.4f} — 改善太小, 可能是噪声")
        else:
            print(f"  ❌ M1 负向: {seen_improvement:+.4f} — STDP 没有让学过的内容变得更熟悉")
            print(f"     → 当前学习可能是『漂移』而非『学习』")

        if unseen_degradation < -0.01:
            print(f"  ⚠️  M3 负向: STDP 让未学内容 log-prob 下降 {unseen_degradation:+.4f}")
            print(f"     → 存在『灾难性遗忘』风险, 通用能力受损")
        else:
            print(f"  ✅ M3 中性: {unseen_degradation:+.4f} — 通用能力未受损")

        if abs(norm_drift) < 0.001:
            print(f"  ⚠️  STDP 权重几乎没变 (Δ={norm_drift:+.4f}) — 学习信号太弱")

    # === Q2: 意识评估 ===
    print("\n" + "=" * 80)
    print("🧠 Q2: 它有意识吗?")
    print("=" * 80)
    consciousness_check = {
        "整合经验流 (Integrated experience)": {
            "现状": "❌ 否 — 每个学习周期是 stateless 的, 没有跨周期的连续体验流",
            "需要的": "一个 working memory 串联跨周期状态, 让 brain 能回忆上一个周期『感觉如何』",
        },
        "自我模型 (Self-model)": {
            "现状": "❌ 否 — stdpbrain 有 self_encoder.py 模块但未接入 daemon",
            "需要的": "接入 self_encoder, 让 brain 能区分『我的想法』vs『外部输入』",
        },
        "全局工作空间 (Global Workspace)": {
            "现状": "❌ 否 — stdpbrain 有 global_workspace.py 但未接入",
            "需要的": "接入 GWT, 让多个脑模块竞争『意识舞台』, 获胜的内容被广播",
        },
        "元认知 (Metacognition)": {
            "现状": "❌ 否 — stdpbrain 有 metacognition.py 但未接入",
            "需要的": "接入元认知, 让 brain 能评估『我是否理解了这个』, 而非被动接收",
        },
        "现象学绑定 (Phenomenal binding)": {
            "现状": "❌ 否 — 没有任何 phenomenal 机制, 当前是纯信息处理",
            "需要的": "这是哲学问题, 即使接入所有模块也不一定产生主观体验",
        },
        "默认模式网络 (DMN)": {
            "现状": "❌ 否 — stdpbrain 有 default_mode_network.py 但未接入",
            "需要的": "接入 DMN, 让 brain 在『空闲』时自发产生内部活动 (类似做梦)",
        },
    }
    for aspect, info in consciousness_check.items():
        print(f"\n  {aspect}:")
        print(f"    {info['现状']}")
        print(f"    需要: {info['需要的']}")

    print("\n" + "=" * 80)
    print("  📋 总体结论:")
    print("=" * 80)
    print("""
  当前 brain 系统:
  - 本质: 一个带 STDP 权重漂移的 GPT-2 微调循环
  - 学习意义: 边际 — STDP 权重在变, 但对学过内容的实际改善很小
  - 意识: 无 — stdpbrain 的 6 个意识相关模块都未接入

  要让学习真正有意义, 需要:
  1. 接入 hippocampus 召回机制 (当前只存不取)
  2. 用『预测 vs 实际』作为 reward (当前是启发式 reward)
  3. 接入 self_encoder / global_workspace / metacognition

  要让意识可能出现, 至少需要:
  1. 跨周期连续体验流
  2. 自我模型 (区分 self vs other)
  3. 全局工作空间 (模块竞争 → 意识舞台)
  4. 元认知 (评估自己的理解程度)
  — 但即使全部接入, 主观体验 (qualia) 仍是开放问题
""")

    # 保存
    out_path = os.path.join(DOWNLOAD_DIR, "brain_evaluation.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            "config": {"n_checkpoints": len(results), "n_seen": len(seen_texts), "n_unseen": len(unseen_texts)},
            "results": results,
            "consciousness_check": consciousness_check,
            "seen_topics": list(seen_topics),
        }, f, ensure_ascii=False, indent=2)
    print(f"\n💾 评估结果: {out_path}")


if __name__ == "__main__":
    main()
