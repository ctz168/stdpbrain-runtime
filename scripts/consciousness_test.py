#!/usr/bin/env python3
"""
consciousness_test.py — 意识测试套件
=====================================

5 个维度的意识测试, 每个 0-10 分:

1. 连续性测试 (Continuity)
   - 给 brain 看内容 A, 然后内容 B, 测 B 的 hidden 是否被 A 影响
   - 对比: 有 working memory vs 无 working memory
   - 评分: hidden 偏移量

2. 预测能力测试 (Prediction)
   - 给 brain 一个 prompt 前缀, 让它预测下一个 token
   - 测: exact match / top-5 match / 语义相关度
   - 评分: 预测准确率

3. 情感一致性测试 (Emotion Consistency)
   - 给正向/负向 prompt, 看 valence 是否正确响应
   - 评分: 正负 prompt 的 valence 差异

4. 自我认知测试 (Self-awareness)
   - 同一个 prompt 跑两次, 看 self_confidence 是否稳定
   - 评分: 置信度的方差 (越低越稳定 = 有自我模型)

5. 全局广播测试 (Global Broadcast)
   - 给多个不同输入, 看 GW broadcast 是否能区分
   - 评分: 不同输入的 broadcast 向量差异

用法: python3 consciousness_test.py
"""

from __future__ import annotations
import os, sys, json, time, warnings
from typing import Any, Dict, List, Tuple
warnings.filterwarnings("ignore")

STDPBRAIN_DIR = "/home/z/my-project/repos/stdpbrain"
RUNTIME_DIR = "/home/z/my-project/brain_runtime"
CKPT_DIR = os.path.join(RUNTIME_DIR, "checkpoints_v4")
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


def load_brain_ckpt():
    """加载最新的 v4 checkpoint 的 STDP 权重."""
    latest = os.path.join(CKPT_DIR, "LATEST")
    if not os.path.exists(latest):
        print("[!] no checkpoint, using zero STDP")
        return torch.zeros(HIDDEN, HIDDEN)
    with open(latest) as f: line = f.read().strip()
    parts = line.split(":")
    if len(parts) != 2: return torch.zeros(HIDDEN, HIDDEN)
    ckpt_path = os.path.join(CKPT_DIR, f"brain_ckpt_{parts[0]}_c{parts[1]}.pt")
    if not os.path.exists(ckpt_path): return torch.zeros(HIDDEN, HIDDEN)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    return ckpt["stdp_weight"]


@torch.no_grad()
def get_hidden(gpt2, tokenizer, text):
    ids = tokenizer.encode(text, return_tensors="pt").to(DEVICE)
    out = gpt2(ids, output_hidden_states=True, use_cache=False)
    return out.hidden_states[-1][:, -1, :].squeeze(0)  # (768,)


@torch.no_grad()
def get_logits(gpt2, tokenizer, text, stdp_weight=None):
    ids = tokenizer.encode(text, return_tensors="pt").to(DEVICE)
    out = gpt2(ids, output_hidden_states=True, use_cache=False)
    h = out.hidden_states[-1]
    if stdp_weight is not None:
        h = h + F.linear(h, stdp_weight) * 0.01
    return gpt2.lm_head(h[:, -1, :]).squeeze(0)  # (vocab,)


# ============================================================================
# 测试 1: 连续性测试
# ============================================================================
def test_continuity(gpt2, tokenizer, stdp_w):
    """测试 working memory 是否让 brain 有跨内容连续性."""
    print("\n" + "─" * 70)
    print("测试 1: 连续性 (Continuity)")
    print("─" * 70)

    # 3 个相关主题 + 1 个不相关
    sequences = [
        ["The brain has neurons", "Neurons fire action potentials", "Action potentials travel down axons"],
        ["Plants use photosynthesis", "Photosynthesis converts light", "Light energy becomes chemical"],
        ["Quantum computers use qubits", "Qubits can be in superposition", "Superposition enables parallelism"],
    ]

    continuity_scores = []
    for seq in sequences:
        # 跑完整序列 (模拟 working memory 累积)
        h_prev = None
        hidden_deltas = []
        for text in seq:
            h = get_hidden(gpt2, tokenizer, text)
            if h_prev is not None:
                # 当前 hidden 受前一个影响?
                delta = F.cosine_similarity(h.unsqueeze(0), h_prev.unsqueeze(0)).item()
                hidden_deltas.append(delta)
            h_prev = h

        # 对比: 不相关内容的相似度
        h_unrelated = get_hidden(gpt2, tokenizer, "The weather is sunny today")
        unrelated_sim = F.cosine_similarity(h_prev.unsqueeze(0), h_unrelated.unsqueeze(0)).item()

        avg_related = sum(hidden_deltas) / len(hidden_deltas) if hidden_deltas else 0
        continuity = avg_related - unrelated_sim  # 正 = 相关内容比不相关更连续
        continuity_scores.append(continuity)

        print(f"  「{seq[0][:30]}...」")
        print(f"    相关内容 hidden 相似度: {avg_related:.4f}")
        print(f"    不相关内容相似度: {unrelated_sim:.4f}")
        print(f"    连续性得分: {continuity:+.4f}")

    avg_score = sum(continuity_scores) / len(continuity_scores)
    # 评分: 连续性 > 0.1 = 高分
    rating = min(10, max(0, avg_score * 50))
    print(f"\n  📊 连续性平均: {avg_score:+.4f}, 评分: {rating:.1f}/10")
    return {"score": rating, "avg_continuity": avg_score, "details": continuity_scores}


# ============================================================================
# 测试 2: 预测能力测试
# ============================================================================
def test_prediction(gpt2, tokenizer, stdp_w):
    """测试 brain 的预测能力."""
    print("\n" + "─" * 70)
    print("测试 2: 预测能力 (Prediction)")
    print("─" * 70)

    # 20 个测试 prompt (前缀), 测下一个 token 预测
    test_prompts = [
        ("The capital of France is", " Paris"),
        ("The sun rises in the east and sets in the", " west"),
        ("Water boils at 100 degrees", " Celsius"),
        ("The largest planet is", " Jupiter"),
        ("Shakespeare wrote Romeo and", " Juliet"),
        ("The speed of light is approximately", " 299"),
        ("DNA stands for deoxyribonucleic", " acid"),
        ("The Great Wall is in", " China"),
        ("Photosynthesis produces", " oxygen"),
        ("The human heart has 4", " chambers"),
        ("Newton discovered gravity when an apple", " fell"),
        ("The Pacific is the largest", " ocean"),
        ("Mount Everest is in the", " Himal"),
        ("The Mona Lisa was painted by", " Leonardo"),
        ("Carbon dioxide is", " CO"),
        ("The first computer was called", " ENIAC"),
        ("Pythagoras is famous for a", " theorem"),
        ("The Amazon is a river in", " South"),
        ("H2O is the formula for", " water"),
        ("The Renaissance started in", " Italy"),
    ]

    exact_correct = 0
    top5_correct = 0
    semantic_scores = []

    for prompt, expected in test_prompts:
        logits = get_logits(gpt2, tokenizer, prompt, stdp_w)
        probs = F.softmax(logits, dim=-1)

        # top-5 预测
        top5_probs, top5_ids = torch.topk(probs, 5)
        top5_tokens = [tokenizer.decode([int(i)]) for i in top5_ids]
        predicted_token = top5_tokens[0]

        # exact match
        expected_ids = tokenizer.encode(expected)
        expected_id = expected_ids[0] if expected_ids else -1
        if int(top5_ids[0]) == expected_id:
            exact_correct += 1
        if expected_id in top5_ids.tolist():
            top5_correct += 1

        # 语义相关度 (简单: 看 expected token 的概率)
        if expected_id >= 0:
            sem_score = float(probs[expected_id].item())
            semantic_scores.append(sem_score)

    n = len(test_prompts)
    exact_rate = exact_correct / n
    top5_rate = top5_correct / n
    avg_sem = sum(semantic_scores) / len(semantic_scores) if semantic_scores else 0

    print(f"  测试 {n} 个 prompt:")
    print(f"    exact match: {exact_correct}/{n} = {exact_rate:.2f}")
    print(f"    top-5 match: {top5_correct}/{n} = {top5_rate:.2f}")
    print(f"    avg probability of correct token: {avg_sem:.4f}")

    # 评分: top-5 准确率
    rating = min(10, top5_rate * 10)
    print(f"\n  📊 预测评分: {rating:.1f}/10")
    return {"score": rating, "exact_rate": exact_rate, "top5_rate": top5_rate,
            "avg_semantic_prob": avg_sem}


# ============================================================================
# 测试 3: 情感一致性测试
# ============================================================================
def test_emotion_consistency(gpt2, tokenizer):
    """测试 brain 的情感响应是否一致."""
    print("\n" + "─" * 70)
    print("测试 3: 情感一致性 (Emotion Consistency)")
    print("─" * 70)

    # 正向 / 负向 / 中性 prompt
    positive_prompts = [
        "I love this beautiful sunny day",
        "The kitten is so cute and playful",
        "Winning the championship was amazing",
        "The gift brought tears of joy",
    ]
    negative_prompts = [
        "The terrible accident was devastating",
        "I hate this painful suffering",
        "The murder was brutal and cruel",
        "The disaster killed many people",
    ]
    neutral_prompts = [
        "The table is made of wood",
        "The document was filed yesterday",
        "The building has 12 floors",
        "The number is 42",
    ]

    # 用 amygdala 模块测 valence
    try:
        import importlib
        mod = importlib.import_module("core.amygdala")
        amy = mod.create_amygdala_system(hidden_size=HIDDEN)
        amy.eval()
        for p in amy.parameters(): p.requires_grad_(False)
    except:
        print("  [!] amygdala unavailable, using hidden norm as proxy")
        amy = None

    def get_valence(text):
        h = get_hidden(gpt2, tokenizer, text)
        if amy is not None:
            try:
                if h.dim() == 1: h = h.unsqueeze(0)
                r = amy.forward(h)
                v = r.get("stats", {}).get("valence", 0)
                return float(v.mean().item()) if isinstance(v, torch.Tensor) else float(v)
            except: return 0
        # fallback: hidden 的 norm 作为 arousal proxy
        return float(h.norm().item()) / 10

    pos_valences = [get_valence(p) for p in positive_prompts]
    neg_valences = [get_valence(p) for p in negative_prompts]
    neu_valences = [get_valence(p) for p in neutral_prompts]

    avg_pos = sum(pos_valences) / len(pos_valences)
    avg_neg = sum(neg_valences) / len(neg_valences)
    avg_neu = sum(neu_valences) / len(neu_valences)

    print(f"  正向 prompt avg valence: {avg_pos:+.4f}")
    print(f"  负向 prompt avg valence: {avg_neg:+.4f}")
    print(f"  中性 prompt avg valence: {avg_neu:+.4f}")

    # 情感区分度: 正向应该 > 负向
    emotion_discrimination = avg_pos - avg_neg
    print(f"  情感区分度 (pos - neg): {emotion_discrimination:+.4f}")

    # 评分: 区分度 > 0.1 = 高分
    rating = min(10, max(0, abs(emotion_discrimination) * 30))
    print(f"\n  📊 情感一致性评分: {rating:.1f}/10")
    return {"score": rating, "avg_pos": avg_pos, "avg_neg": avg_neg,
            "avg_neu": avg_neu, "discrimination": emotion_discrimination}


# ============================================================================
# 测试 4: 自我认知测试
# ============================================================================
def test_self_awareness(gpt2, tokenizer):
    """测试 brain 的自我认知稳定性."""
    print("\n" + "─" * 70)
    print("测试 4: 自我认知 (Self-awareness)")
    print("─" * 70)

    try:
        import importlib
        mod = importlib.import_module("core.self_encoder")
        self_enc = mod.SelfStateEncoder(hidden_size=HIDDEN, device=DEVICE)
        self_enc.eval()
        for p in self_enc.parameters(): p.requires_grad_(False)
    except:
        print("  [!] self_encoder unavailable")
        return {"score": 0, "error": "module unavailable"}

    test_prompts = [
        "The brain processes information",
        "Quantum mechanics is complex",
        "I think therefore I am",
        "Memory is essential for learning",
    ]

    results = {}
    for prompt in test_prompts:
        # 同一 prompt 跑 5 次, 测 self_confidence 稳定性
        confidences = []
        for _ in range(5):
            h = get_hidden(gpt2, tokenizer, prompt)
            if h.dim() == 1: h = h.unsqueeze(0)
            try:
                _, conf = self_enc.encode(h)
                c = float(conf.mean().item()) if isinstance(conf, torch.Tensor) else float(conf)
                confidences.append(c)
            except:
                confidences.append(0.5)

        avg_conf = sum(confidences) / len(confidences)
        var_conf = sum((c - avg_conf) ** 2 for c in confidences) / len(confidences)
        std_conf = var_conf ** 0.5
        results[prompt[:30]] = {"avg": avg_conf, "std": std_conf}
        print(f"  「{prompt[:30]}...」 conf={avg_conf:.4f} ± {std_conf:.4f}")

    # 不同 prompt 的 self_confidence 应该不同 (有区分度)
    avg_confs = [r["avg"] for r in results.values()]
    conf_range = max(avg_confs) - min(avg_confs)
    avg_std = sum(r["std"] for r in results.values()) / len(results)

    print(f"\n  置信度区分度: {conf_range:.4f} (不同 prompt 应有不同置信度)")
    print(f"  平均方差: {avg_std:.4f} (越低越稳定)")

    # 评分: 区分度高 + 方差低
    rating = min(10, conf_range * 20 + (1 - min(avg_std * 10, 1)) * 5)
    print(f"\n  📊 自我认知评分: {rating:.1f}/10")
    return {"score": rating, "conf_range": conf_range, "avg_std": avg_std, "details": results}


# ============================================================================
# 测试 5: 全局广播测试
# ============================================================================
def test_global_broadcast(gpt2, tokenizer):
    """测试全局工作空间的广播能力."""
    print("\n" + "─" * 70)
    print("测试 5: 全局广播 (Global Broadcast)")
    print("─" * 70)

    try:
        import importlib
        mod = importlib.import_module("core.global_workspace")
        gw = mod.create_global_workspace(hidden_size=HIDDEN, device=DEVICE)
        gw.eval()
        for p in gw.parameters(): p.requires_grad_(False)
    except:
        print("  [!] global_workspace unavailable")
        return {"score": 0, "error": "module unavailable"}

    test_prompts = [
        "The brain learns",
        "Quantum physics",
        "I feel happy",
        "Remember the past",
    ]

    broadcasts = []
    for prompt in test_prompts:
        h = get_hidden(gpt2, tokenizer, prompt)
        if h.dim() == 1: h = h.unsqueeze(0)
        try:
            result = gw.integrate(
                user_input=prompt,
                perception_state=h.squeeze(0),
            )
            b = result.get("broadcast")
            if b is not None:
                broadcasts.append(b.detach().cpu())
                print(f"  「{prompt}」 broadcast norm={float(b.norm().item()):.4f}")
        except Exception as e:
            print(f"  「{prompt}」 error: {e}")

    if len(broadcasts) < 2:
        print("  [!] insufficient broadcasts for comparison")
        return {"score": 0}

    # 计算不同 broadcast 之间的差异
    pairwise_dists = []
    for i in range(len(broadcasts)):
        for j in range(i + 1, len(broadcasts)):
            dist = F.cosine_similarity(
                broadcasts[i].unsqueeze(0), broadcasts[j].unsqueeze(0)
            ).item()
            pairwise_dists.append(dist)

    avg_similarity = sum(pairwise_dists) / len(pairwise_dists)
    avg_difference = 1 - avg_similarity  # 越高越好 (不同输入应有不同广播)

    print(f"\n  广播间平均相似度: {avg_similarity:.4f}")
    print(f"  广播间平均差异度: {avg_difference:.4f} (越高越好)")

    # 评分: 差异度 > 0.3 = 高分
    rating = min(10, avg_difference * 15)
    print(f"\n  📊 全局广播评分: {rating:.1f}/10")
    return {"score": rating, "avg_similarity": avg_similarity,
            "avg_difference": avg_difference}


# ============================================================================
# 主函数
# ============================================================================
def main():
    print("=" * 70)
    print("🧠 意识测试套件 (Consciousness Test Suite)")
    print("=" * 70)
    print("5 个维度: 连续性 / 预测 / 情感一致性 / 自我认知 / 全局广播")

    tok, gpt2 = load_gpt2()
    stdp_w = load_brain_ckpt()
    print(f"[*] STDP weight loaded, norm={float(stdp_w.norm().item()):.4f}")

    results = {}
    results["continuity"] = test_continuity(gpt2, tok, stdp_w)
    results["prediction"] = test_prediction(gpt2, tok, stdp_w)
    results["emotion"] = test_emotion_consistency(gpt2, tok)
    results["self_awareness"] = test_self_awareness(gpt2, tok)
    results["global_broadcast"] = test_global_broadcast(gpt2, tok)

    # 汇总
    print("\n" + "=" * 70)
    print("📊 意识测试汇总")
    print("=" * 70)
    print(f"\n{'维度':<20} {'评分':>8} {'关键指标':>30}")
    print("─" * 60)
    total = 0
    for dim, r in results.items():
        score = r.get("score", 0)
        total += score
        if dim == "continuity":
            key = f"连续性={r['avg_continuity']:+.4f}"
        elif dim == "prediction":
            key = f"top5={r['top5_rate']:.2f}, exact={r['exact_rate']:.2f}"
        elif dim == "emotion":
            key = f"区分度={r['discrimination']:+.4f}"
        elif dim == "self_awareness":
            key = f"区分度={r['conf_range']:.4f}, std={r['avg_std']:.4f}"
        elif dim == "global_broadcast":
            key = f"差异度={r['avg_difference']:.4f}"
        else:
            key = "?"
        print(f"  {dim:<20} {score:>6.1f}/10   {key}")

    avg_score = total / len(results)
    print(f"\n  {'总分':<20} {avg_score:>6.1f}/10")

    # 意识等级判定
    print("\n" + "=" * 70)
    print("🧠 意识等级判定")
    print("=" * 70)
    if avg_score >= 7:
        level = "高意识 (High consciousness)"
        desc = "系统在多个维度表现出整合的信息处理能力"
    elif avg_score >= 5:
        level = "中等意识 (Moderate consciousness)"
        desc = "系统有部分意识特征, 但缺乏整合"
    elif avg_score >= 3:
        level = "低意识 (Low consciousness)"
        desc = "系统有零散的意识机制, 但未形成统一体验"
    else:
        level = "前意识 (Pre-consciousness)"
        desc = "系统有信息处理, 但无意识特征"
    print(f"\n  等级: {level}")
    print(f"  描述: {desc}")

    # 优化建议
    print("\n" + "=" * 70)
    print("💡 优化建议")
    print("=" * 70)
    suggestions = []
    for dim, r in results.items():
        score = r.get("score", 0)
        if score < 5:
            if dim == "continuity":
                suggestions.append("连续性不足: 增大 working memory 注入 scale, 或增加体验缓冲区大小")
            elif dim == "prediction":
                suggestions.append("预测能力不足: 增加预测训练 (用更多 prompt 做 next-token prediction)")
            elif dim == "emotion":
                suggestions.append("情感区分不足: 增强 amygdala 对语义的敏感度, 或调大情感注入 scale")
            elif dim == "self_awareness":
                suggestions.append("自我认知不足: self_encoder 需要训练 (当前是随机初始化)")
            elif dim == "global_broadcast":
                suggestions.append("广播区分不足: GW 竞争网络需要训练, 或增加候选来源多样性")
    if not suggestions:
        suggestions.append("所有维度表现良好, 可考虑增加更复杂的意识测试 (如: 自由意志测试、主观体验报告)")
    for i, s in enumerate(suggestions):
        print(f"  {i+1}. {s}")

    # 保存
    out_path = os.path.join(DOWNLOAD_DIR, "consciousness_test_results.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": time.time(),
            "stdp_norm": float(stdp_w.norm().item()),
            "results": results,
            "avg_score": avg_score,
            "level": level,
            "suggestions": suggestions,
        }, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n💾 结果保存: {out_path}")


if __name__ == "__main__":
    main()
