#!/usr/bin/env python3
"""
GPT-2 × stdpbrain v4 — 自主学习系统 (接入网络)
==============================================

让 brain 系统真正"上网学习":
1. CuriosityEngine: 基于"好奇心"生成 query (从主题池 + 随机组合)
2. WebFetcher: 用 z-ai CLI (web_search + page_reader) 搜索并抓取网页
3. ContentProcessor: HTML → 纯文本 → 切成训练 prompt
4. STDPBrainV3 (复用): 用 STDP 三因子 + 外部 reward 训练
5. HippocampusLite: 把学到的内容存为"记忆"
6. ThoughtStream (复用 v2): 生成"学习反思"思维流

主循环 (5 个学习周期):
  好奇心 → 搜索 → 阅读 → 切片 → STDP 训练 → 存记忆 → 反思

输出:
  /home/z/my-project/download/gpt2_stdpbrain_v4_autolearn.json
  /home/z/my-project/download/gpt2_stdpbrain_v4_autolearn.txt
  /home/z/my-project/download/gpt2_stdpbrain_v4_journal.md   (学习日志)
  /home/z/my-project/download/gpt2_stdpbrain_v4.png
"""

from __future__ import annotations

import os, sys, json, time, random, subprocess, re, warnings, hashlib
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass, field

warnings.filterwarnings("ignore")

STDPBRAIN_DIR = "/home/z/my-project/repos/stdpbrain"
DOWNLOAD_DIR = "/home/z/my-project/download"
SCRIPTS_DIR = "/home/z/my-project/scripts"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.chdir(STDPBRAIN_DIR)
sys.path.insert(0, STDPBRAIN_DIR)
sys.path.insert(0, SCRIPTS_DIR)

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

# === 学习参数 ===
N_LEARNING_CYCLES = 5        # 5 个学习周期
PROMPTS_PER_CYCLE = 4         # 每周期从网页提取 4 个训练 prompt
MAX_NEW_TOKENS = 8            # 每 prompt 生成 8 个 token
STDP_LR = 1e-3
STDP_WEIGHT_CLAMP = 0.3

# === 好奇心主题池 ===
# brain 系统的"兴趣"——覆盖科学/技术/人文/自然
CURIOSITY_TOPICS = [
    "neuroplasticity brain learning",
    "quantum computing explained",
    "photosynthesis how plants work",
    "history of artificial intelligence",
    "ocean deep sea creatures",
    "how vaccines work immune system",
    "black holes spacetime physics",
    "DNA CRISPR gene editing",
    "climate change carbon cycle",
    "ancient Egyptian civilization",
    "how memory works hippocampus",
    "volcanoes plate tectonics",
    "machine learning neural networks",
    "renaissance art history",
    "symbiosis in nature",
    "how sleep affects the brain",
]


# ============================================================================
# 1. CuriosityEngine — 好奇心引擎
# ============================================================================
class CuriosityEngine:
    """基于"好奇心"生成学习主题.

    策略:
    - 70% 从主题池随机选
    - 30% 基于已学内容做"关联探索" (从一个学过的关键词跳到相关的)
    """

    def __init__(self):
        self.learned_topics: List[str] = []
        self.learned_keywords: List[str] = []
        self.cycle_count = 0

    def generate_query(self) -> Tuple[str, str]:
        """返回 (query, reason) — reason 是思维流解释为什么选这个主题"""
        self.cycle_count += 1

        if random.random() < 0.3 and self.learned_keywords:
            # 关联探索: 从已学关键词跳到相关的
            seed_kw = random.choice(self.learned_keywords)
            # 简单的关联规则
            associations = {
                "brain": ["consciousness", "neurons", "memory"],
                "neuroplasticity": ["learning", "stroke recovery", "meditation"],
                "quantum": ["entanglement", "uncertainty principle"],
                "DNA": ["RNA", "proteins", "genetics"],
                "climate": ["global warming", "oceans", "weather"],
                "AI": ["machine learning", "neural networks", "deep learning"],
                "memory": ["hippocampus", "sleep", "dreams"],
            }
            related = associations.get(seed_kw, [])
            if related:
                topic = random.choice(related)
                query = f"{seed_kw} {topic}"
                reason = f"[curiosity] 刚学了「{seed_kw}」, 想顺着关联到「{topic}」——知识网络扩张"
                return query, reason

        # 默认: 从主题池随机选
        topic = random.choice(CURIOSITY_TOPICS)
        reason = f"[curiosity] 第 {self.cycle_count} 个学习周期, 随机探索「{topic}」"
        return topic, reason

    def record_learning(self, topic: str, keywords: List[str]):
        self.learned_topics.append(topic)
        self.learned_keywords.extend(keywords)


# ============================================================================
# 2. WebFetcher — 网络搜索 + 网页阅读
# ============================================================================
class WebFetcher:
    """用 z-ai CLI 搜索并抓取网页内容."""

    def __init__(self):
        self.search_cache: Dict[str, List[Dict]] = {}
        self.page_cache: Dict[str, str] = {}

    def search(self, query: str, num: int = 5) -> List[Dict]:
        """搜索网页, 返回结果列表."""
        if query in self.search_cache:
            return self.search_cache[query]

        cache_file = f"/tmp/stdpbrain_search_{hashlib.md5(query.encode()).hexdigest()[:8]}.json"
        try:
            result = subprocess.run(
                ["z-ai", "function", "-n", "web_search",
                 "-a", json.dumps({"query": query, "num": num}),
                 "-o", cache_file],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode != 0:
                print(f"    [search] CLI failed: {result.stderr[:100]}", flush=True)
                return []

            with open(cache_file) as f:
                data = json.load(f)
            results = data if isinstance(data, list) else data.get("data", [])
            self.search_cache[query] = results
            return results
        except Exception as e:
            print(f"    [search] error: {e}", flush=True)
            return []

    def read_page(self, url: str) -> Optional[Dict]:
        """读取一个网页, 返回 {title, text, url}."""
        if url in self.page_cache:
            return {"title": "(cached)", "text": self.page_cache[url], "url": url}

        cache_file = f"/tmp/stdpbrain_page_{hashlib.md5(url.encode()).hexdigest()[:8]}.json"
        try:
            result = subprocess.run(
                ["z-ai", "function", "-n", "page_reader",
                 "-a", json.dumps({"url": url}),
                 "-o", cache_file],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode != 0:
                print(f"    [page_reader] CLI failed: {result.stderr[:100]}", flush=True)
                return None

            with open(cache_file) as f:
                data = json.load(f)

            inner = data.get("data", data)
            title = inner.get("title", "")
            html = inner.get("html", "")

            # HTML → 纯文本
            text = re.sub(r'<script[^>]*>[\s\S]*?</script>', '', html, flags=re.I)
            text = re.sub(r'<style[^>]*>[\s\S]*?</style>', '', text, flags=re.I)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()

            if len(text) < 100:
                return None

            self.page_cache[url] = text
            return {"title": title, "text": text, "url": url}
        except Exception as e:
            print(f"    [page_reader] error: {e}", flush=True)
            return None


# ============================================================================
# 3. ContentProcessor — 把网页文本切成训练 prompt
# ============================================================================
class ContentProcessor:
    """把网页纯文本切成 GPT-2 训练 prompt."""

    @staticmethod
    def extract_prompts(text: str, n: int = PROMPTS_PER_CYCLE) -> List[str]:
        """从文本中提取 n 个 prompt (每个 30-80 字符的完整句子开头)."""
        # 按句号/问号/感叹号切分
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # 筛选: 长度 30-200, 含字母
        candidates = []
        for s in sentences:
            s = s.strip()
            if 30 < len(s) < 200 and re.search(r'[a-zA-Z]{3,}', s):
                # 取前 60-80 字符作为 prompt
                if len(s) > 80:
                    # 找一个合适的截断点 (空格)
                    cut = s[:80].rfind(' ')
                    if cut > 30:
                        s = s[:cut]
                candidates.append(s)

        # 去重 + 打乱
        seen = set()
        unique = []
        for s in candidates:
            if s not in seen:
                seen.add(s)
                unique.append(s)

        random.shuffle(unique)
        return unique[:n]

    @staticmethod
    def extract_keywords(text: str, topk: int = 5) -> List[str]:
        """简单关键词提取 (基于词频)."""
        # 移除常见停用词
        stop = set("the a an and or but in on at to for of is are was were be been being have has had do does did will would could should may might must can this that these those it its their there here as by with from into out up down over under again further then once".split())
        words = re.findall(r'[a-zA-Z]{4,}', text.lower())
        word_freq = {}
        for w in words:
            if w not in stop:
                word_freq[w] = word_freq.get(w, 0) + 1
        sorted_words = sorted(word_freq.items(), key=lambda x: -x[1])
        return [w for w, _ in sorted_words[:topk]]


# ============================================================================
# 4. HippocampusLite — 简化版海马体 (存学到的记忆)
# ============================================================================
@dataclass
class Memory:
    """一条情景记忆."""
    id: str
    timestamp: float
    source_url: str
    source_title: str
    topic: str
    keywords: List[str]
    prompts: List[str]
    reward: float
    stdp_changed_pct: float
    reflection: str  # 学习反思


class HippocampusLite:
    """简化版海马体 — 存学到的内容, 支持召回."""

    def __init__(self):
        self.memories: List[Memory] = []

    def store(self, mem: Memory):
        self.memories.append(mem)
        print(f"    [hippocampus] stored memory #{len(self.memories)}: "
              f"topic={mem.topic!r} keywords={mem.keywords[:3]}", flush=True)

    def recall_by_keyword(self, keyword: str, topk: int = 3) -> List[Memory]:
        """根据关键词召回记忆."""
        scored = []
        for m in self.memories:
            score = sum(1 for k in m.keywords if keyword.lower() in k.lower())
            if score > 0:
                scored.append((score, m))
        scored.sort(key=lambda x: -x[0])
        return [m for _, m in scored[:topk]]

    def get_summary(self) -> Dict:
        if not self.memories:
            return {"total": 0}
        return {
            "total": len(self.memories),
            "topics": [m.topic for m in self.memories],
            "avg_reward": sum(m.reward for m in self.memories) / len(self.memories),
            "avg_stdp_changed": sum(m.stdp_changed_pct for m in self.memories) / len(self.memories),
            "all_keywords": list(set(kw for m in self.memories for kw in m.keywords)),
        }


# ============================================================================
# 5. ThoughtStream (复用 v2 的设计)
# ============================================================================
class ThoughtStream:
    """学习过程中的思维流."""

    STATES = ["FOCUSED", "WANDERING", "REFLECTING", "RESTING"]

    LEADS = {
        "FOCUSED":    ["[focus] 让我聚焦分析这个新知识...", "[focus] 仔细看看这段内容..."],
        "WANDERING":  ["[wander] 这让我联想到...", "[wander] 顺便想到..."],
        "REFLECTING": ["[reflect] 我刚学到了什么？让我整理一下...", "[reflect] 重新审视..."],
        "RESTING":    ["[rest] 嗯, 整理一下记忆...", "[rest] 消化刚才学到的..."],
    }

    def __init__(self):
        self.state = "FOCUSED"
        self.history: List[Dict[str, Any]] = []

    def transition(self, da: float, ne: float, valence: float):
        # 简化: 高 DA → FOCUSED, 低 NE → RESTING, 负 valence → REFLECTING
        if da > 0.55:
            self.state = "FOCUSED"
        elif ne < 0.3:
            self.state = "RESTING"
        elif valence < -0.05:
            self.state = "REFLECTING"
        else:
            self.state = random.choice(["WANDERING", "FOCUSED"])

    def generate_reflection(
        self, topic: str, prompts: List[str], brain_snap: Dict[str, float],
        stdp_changed_pct: float, avg_reward: float,
    ) -> str:
        """生成一段学习反思."""
        lead = random.choice(self.LEADS[self.state])
        # 取学到的第一个 prompt 作为代表
        sample = prompts[0] if prompts else "(空)"
        reflection = (
            f"{lead} 刚学了「{topic}」, "
            f"读到一句「{sample[:50]}...」. "
            f"训练了 {len(prompts)} 个 prompt, "
            f"STDP 改变了 {stdp_changed_pct:.0f}% 的 token, "
            f"平均 reward={avg_reward:.3f}. "
            f"DA={brain_snap.get('da', 0.5):.3f}, "
            f"NE={brain_snap.get('ne', 0.5):.3f}, "
            f"valence={brain_snap.get('valence', 0):+.3f}. "
        )
        if stdp_changed_pct > 40:
            reflection += "STDP 在大量改变输出——说明这个新知识真的在影响我的'思考'. "
        elif stdp_changed_pct > 20:
            reflection += "STDP 部分改变输出——我在部分吸收这个新知识. "
        else:
            reflection += "STDP 改变较少——可能这个领域我已经比较熟悉. "

        if brain_snap.get('valence', 0) > 0.05:
            reflection += "杏仁核判定这个学习是正向的, 情绪偏积极. "
        elif brain_snap.get('valence', 0) < -0.05:
            reflection += "杏仁核判定情绪偏消极——也许是陌生或困难的概念. "

        self.history.append({
            "state": self.state,
            "topic": topic,
            "reflection": reflection,
            "brain": {k: v for k, v in brain_snap.items()},
        })
        return reflection


# ============================================================================
# 6. STDP Brain v4 (复用 v3 的核心, 简化)
# ============================================================================
class STDPBrainV4:
    """STDP brain — 简化版, 用于自主学习."""

    def __init__(self):
        import importlib
        print("[*] Loading brain modules...", flush=True)
        self.modules: Dict[str, nn.Module] = {}
        spec = [
            ("cerebellar",     "core.cerebellar_correction_677",    "create_cerebellar_correction_system",  dict(hidden_size=HIDDEN_SIZE)),
            ("basal_ganglia",  "core.basal_ganglia_dopamine",       "create_basal_ganglia_dopamine_system", dict(hidden_size=HIDDEN_SIZE)),
            ("lc_ne",          "core.locus_coeruleus_ne",           "create_locus_coeruleus_ne_system",     dict(hidden_size=HIDDEN_SIZE)),
            ("amygdala",       "core.amygdala",                     "create_amygdala_system",               dict(hidden_size=HIDDEN_SIZE)),
            ("dual_process",   "core.dual_process",                 "create_dual_process_system",           dict(hidden_size=HIDDEN_SIZE)),
            ("synaptic_plast", "core.synaptic_plasticity",          "create_synaptic_plasticity_system",    dict(hidden_size=HIDDEN_SIZE)),
        ]
        for name, path, factory, kwargs in spec:
            try:
                mod = importlib.import_module(path)
                m = getattr(mod, factory)(**kwargs)
                m.eval() if hasattr(m, "eval") else None
                for p in m.parameters():
                    p.requires_grad_(False)
                self.modules[name] = m
                n = sum(p.numel() for p in m.parameters())
                print(f"    {name:<16} | {n:>10,} params", flush=True)
            except Exception as e:
                print(f"    {name:<16} | ERROR: {e}", flush=True)

        # STDP 突触矩阵
        self.stdp_weight = nn.Parameter(
            torch.randn(HIDDEN_SIZE, HIDDEN_SIZE, device=DEVICE) * 0.01,
            requires_grad=True,
        )
        self.optimizer = torch.optim.SGD([self.stdp_weight], lr=STDP_LR, momentum=0.9, weight_decay=1e-4)

        # 找 sp 的 a_plus/a_minus 用于元更新
        sp = self.modules.get("synaptic_plast")
        self.sp_meta_params: Dict[str, Any] = {}
        if sp is not None:
            for sub_name, sub_mod in sp.named_modules():
                for attr in ['a_plus', 'a_minus']:
                    if hasattr(sub_mod, attr):
                        self.sp_meta_params[f"{sub_name}.{attr}"] = getattr(sub_mod, attr)

        self.da_level = 0.5
        self.ne_level = 0.5
        self.valence = 0.0
        self.initial_stdp_norm = float(self.stdp_weight.norm().item())
        print(f"    STDP_W: 768×768 (initial norm={self.initial_stdp_norm:.4f})", flush=True)

    def step(self, hidden_now, hidden_prev, external_reward):
        snap = {}
        H = hidden_now.detach()

        with torch.no_grad():
            stdp_delta = F.linear(H, self.stdp_weight)
            modified_hidden = H + stdp_delta * 0.1

            # Brain 闭环
            cb = self.modules.get("cerebellar")
            cerebellar_err = 0.0
            corrected = H
            if cb is not None:
                try:
                    r = cb.forward(H, hidden_prev, H)
                    err = r.get("prediction_error", None)
                    if isinstance(err, torch.Tensor):
                        cerebellar_err = float(err.norm().item())
                    corrected = r.get("corrected_output", H)
                    if not isinstance(corrected, torch.Tensor):
                        corrected = H
                except Exception:
                    pass
            snap["cerebellar_error"] = cerebellar_err

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

            dp = self.modules.get("dual_process")
            system_mode = "?"
            if dp is not None:
                try:
                    r = dp.forward(amy_out)
                    system_mode = str(r.get("stats", {}).get("active_system", "?"))
                except Exception:
                    pass
            snap["system"] = system_mode

            # SP ΔW 信号
            sp = self.modules.get("synaptic_plast")
            delta_w = None
            if sp is not None:
                try:
                    new_w = sp.forward(
                        pre_activity=hidden_prev,
                        post_activity=H,
                        dopamine_level=da_level,
                        weights=self.stdp_weight.data.clone(),
                    )
                    delta_w = (new_w - self.stdp_weight.data).detach()
                except Exception:
                    delta_w = None

        combined_reward = 0.5 * external_reward + 0.5 * (da_level - 0.5)
        snap["external_reward"] = external_reward
        snap["combined_reward"] = combined_reward

        if delta_w is not None:
            with torch.no_grad():
                pseudo_grad = -delta_w * combined_reward * 0.1
                if self.stdp_weight.grad is None:
                    self.stdp_weight.grad = pseudo_grad.clone()
                else:
                    self.stdp_weight.grad.copy_(pseudo_grad)

                # 元更新 a_plus / a_minus
                meta_lr = 0.0005
                for name, param in self.sp_meta_params.items():
                    if "a_plus" in name.lower():
                        if isinstance(param, torch.Tensor):
                            param.add_(combined_reward * meta_lr)
                            param.clamp_(0.0001, 0.1)
                        else:
                            new_val = float(param) + combined_reward * meta_lr
                            new_val = max(0.0001, min(0.1, new_val))
                            parts = name.split(".")
                            obj = self.modules["synaptic_plast"]
                            for p in parts[:-1]:
                                if p: obj = getattr(obj, p)
                            setattr(obj, parts[-1], new_val)
                            self.sp_meta_params[name] = new_val
                    elif "a_minus" in name.lower():
                        if isinstance(param, torch.Tensor):
                            param.sub_(combined_reward * meta_lr)
                            param.clamp_(-0.1, -0.0001)
                        else:
                            new_val = float(param) - combined_reward * meta_lr
                            new_val = max(-0.1, min(-0.0001, new_val))
                            parts = name.split(".")
                            obj = self.modules["synaptic_plast"]
                            for p in parts[:-1]:
                                if p: obj = getattr(obj, p)
                            setattr(obj, parts[-1], new_val)
                            self.sp_meta_params[name] = new_val

        snap["stdp_norm"] = float(self.stdp_weight.norm().item())
        snap["stdp_grad_norm"] = float(self.stdp_weight.grad.norm().item()) if self.stdp_weight.grad is not None else 0.0

        if self.stdp_weight.grad is not None:
            gn = self.stdp_weight.grad.norm().item()
            if gn > 10.0:
                self.stdp_weight.grad.mul_(10.0 / gn)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.stdp_weight.data.clamp_(-STDP_WEIGHT_CLAMP, STDP_WEIGHT_CLAMP)

        return snap, modified_hidden.detach()


# ============================================================================
# 7. 外部 reward (复用 v3)
# ============================================================================
def compute_external_reward(logits, chosen_id, base_id, generated_ids):
    with torch.no_grad():
        probs = F.softmax(logits, dim=-1)
        confidence = float(probs[chosen_id].item())
        recent = generated_ids[-5:] if len(generated_ids) >= 5 else generated_ids
        rep_count = sum(1 for t in recent if t == chosen_id)
        rep_penalty = -0.3 * rep_count
        if chosen_id != base_id:
            stdp_bonus = +0.2 if confidence > 0.05 and rep_count == 0 else -0.1
        else:
            stdp_bonus = 0.0
        log_probs = torch.log(probs + 1e-8)
        entropy = -float((probs * log_probs).sum().item())
        entropy_bonus = 0.05 * min(entropy, 5.0) / 5.0
    return 0.4 * confidence + 0.3 * rep_penalty + 0.2 * stdp_bonus + 0.1 * entropy_bonus


# ============================================================================
# 8. 训练一个 prompt (复用 v3 逻辑)
# ============================================================================
def train_one_prompt(tokenizer, gpt2, brain: STDPBrainV4, prompt: str, max_new_tokens: int):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    generated_ids = input_ids.clone()

    with torch.no_grad():
        out0 = gpt2(input_ids, output_hidden_states=True, use_cache=False)
        hidden_prev = out0.hidden_states[-1][:, -1, :].squeeze(0).detach()
        if hidden_prev.dim() == 1:
            hidden_prev = hidden_prev.unsqueeze(0)

    snapshots = []
    rewards = []
    stdp_changed_count = 0

    for step in range(max_new_tokens):
        with torch.no_grad():
            out = gpt2(generated_ids, output_hidden_states=True, use_cache=False)
            hidden_now = out.hidden_states[-1][:, -1, :].squeeze(0).detach()
            if hidden_now.dim() == 1:
                hidden_now = hidden_now.unsqueeze(0)
            base_logits = gpt2.lm_head(hidden_now).squeeze(0)
            base_id = int(torch.argmax(base_logits, dim=-1).item())

        ext_reward = rewards[-1] if rewards else 0.5
        snap, modified_hidden = brain.step(hidden_now, hidden_prev, ext_reward)

        with torch.no_grad():
            modified_logits = gpt2.lm_head(modified_hidden).squeeze(0)
            next_id = int(torch.argmax(modified_logits, dim=-1).item())

        gen_ids_list = generated_ids[0].tolist()
        actual_reward = compute_external_reward(modified_logits, next_id, base_id, gen_ids_list)
        rewards.append(actual_reward)

        snap["token"] = tokenizer.decode([next_id])
        snap["token_id"] = next_id
        snap["base_id"] = base_id
        snap["stdp_changed"] = (next_id != base_id)
        if snap["stdp_changed"]:
            stdp_changed_count += 1
        snapshots.append(snap)

        generated_ids = torch.cat([generated_ids, torch.tensor([[next_id]], device=DEVICE)], dim=1)
        hidden_prev = hidden_now.clone()

        if next_id == tokenizer.eos_token_id:
            break

    return {
        "prompt": prompt,
        "generated_text": prompt + "".join(s["token"] for s in snapshots),
        "snapshots": snapshots,
        "avg_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "stdp_changed_pct": stdp_changed_count / len(snapshots) * 100 if snapshots else 0,
        "n_tokens": len(snapshots),
        "last_brain_snap": snapshots[-1] if snapshots else {},
    }


# ============================================================================
# 9. 自主学习主循环
# ============================================================================
def main():
    print("=" * 90, flush=True)
    print("🧠 GPT-2 × stdpbrain v4 — 自主学习系统 (接入网络)", flush=True)
    print("=" * 90, flush=True)
    print(f"Config: {N_LEARNING_CYCLES} cycles × {PROMPTS_PER_CYCLE} prompts × {MAX_NEW_TOKENS} tokens", flush=True)
    print(f"device={DEVICE}, torch={torch.__version__}, threads={TORCH_THREADS}\n", flush=True)

    # 加载 GPT-2
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    print("[1/4] Loading GPT-2...", flush=True)
    t0 = time.time()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt2.eval()
    gpt2.to(DEVICE)
    for p in gpt2.parameters():
        p.requires_grad_(False)
    print(f"    ✅ {sum(p.numel() for p in gpt2.parameters())/1e6:.1f}M params | {time.time()-t0:.1f}s\n", flush=True)

    # 加载 brain
    print("[2/4] Loading STDP brain v4...", flush=True)
    brain = STDPBrainV4()

    # 初始化子系统
    curiosity = CuriosityEngine()
    fetcher = WebFetcher()
    hippocampus = HippocampusLite()
    thoughts = ThoughtStream()

    print(f"\n[3/4] 开始自主学习 ({N_LEARNING_CYCLES} cycles)...\n", flush=True)

    learning_log: List[Dict[str, Any]] = []
    t_start = time.time()

    for cycle in range(N_LEARNING_CYCLES):
        print(f"\n{'─' * 90}", flush=True)
        print(f"📚 学习周期 {cycle+1}/{N_LEARNING_CYCLES}", flush=True)
        print(f"{'─' * 90}", flush=True)

        # 1. 好奇心生成 query
        query, reason = curiosity.generate_query()
        print(f"  💭 {reason}", flush=True)
        print(f"  🔍 搜索: {query!r}", flush=True)

        # 2. 网络搜索
        t0 = time.time()
        search_results = fetcher.search(query, num=5)
        print(f"  📋 找到 {len(search_results)} 个结果 ({time.time()-t0:.1f}s)", flush=True)
        if not search_results:
            print(f"  ⚠️ 无搜索结果, 跳过", flush=True)
            continue

        # 显示 top 3
        for i, r in enumerate(search_results[:3]):
            print(f"     [{i+1}] {r.get('name','?')[:60]} | {r.get('host_name','?')}", flush=True)

        # 3. 选最佳结果阅读 (优先 wikipedia / 教育类)
        best_url = None
        for r in search_results:
            host = r.get('host_name', '')
            if any(d in host for d in ['wikipedia', 'britannica', 'nature', 'sciencedaily', 'ncbi']):
                best_url = r.get('url')
                best_title = r.get('name', '')
                break
        if not best_url:
            best_url = search_results[0].get('url')
            best_title = search_results[0].get('name', '')

        print(f"  📖 阅读: {best_title[:60]}", flush=True)
        print(f"     URL: {best_url}", flush=True)

        t0 = time.time()
        page = fetcher.read_page(best_url)
        if not page:
            print(f"  ⚠️ 阅读失败, 跳过", flush=True)
            continue
        print(f"  📄 抓取到 {len(page['text'])} 字符 ({time.time()-t0:.1f}s)", flush=True)

        # 4. 切片成训练 prompt
        prompts = ContentProcessor.extract_prompts(page['text'], n=PROMPTS_PER_CYCLE)
        keywords = ContentProcessor.extract_keywords(page['text'], topk=5)
        print(f"  ✂️ 切片: {len(prompts)} 个训练 prompt, 关键词: {keywords}", flush=True)
        for i, p in enumerate(prompts):
            print(f"     [{i+1}] {p[:70]!r}", flush=True)

        # 5. STDP 训练
        print(f"  🧠 STDP 训练...", flush=True)
        cycle_results = []
        cycle_stdp_first = brain.stdp_weight.norm().item()
        for i, prompt in enumerate(prompts):
            t0 = time.time()
            result = train_one_prompt(tokenizer, gpt2, brain, prompt, MAX_NEW_TOKENS)
            result["prompt_idx"] = i
            result["time_s"] = time.time() - t0
            cycle_results.append(result)
            chg = result["stdp_changed_pct"]
            r = result["avg_reward"]
            print(f"     [{i+1}] R={r:+.3f} chg={chg:.0f}% | "
                  f"→ {result['generated_text'][:60]!r}", flush=True)

        cycle_stdp_last = brain.stdp_weight.norm().item()
        avg_reward = sum(r["avg_reward"] for r in cycle_results) / len(cycle_results)
        avg_chg = sum(r["stdp_changed_pct"] for r in cycle_results) / len(cycle_results)
        last_snap = cycle_results[-1]["last_brain_snap"]

        print(f"\n  📊 周期总结:", flush=True)
        print(f"     STDP norm: {cycle_stdp_first:.4f} → {cycle_stdp_last:.4f} "
              f"({(cycle_stdp_last-cycle_stdp_first)/max(cycle_stdp_first,1e-8)*100:+.3f}%)", flush=True)
        print(f"     Avg reward: {avg_reward:.4f}", flush=True)
        print(f"     Avg STDP-change: {avg_chg:.1f}%", flush=True)

        # 6. 思维流反思
        thoughts.transition(brain.da_level, brain.ne_level, brain.valence)
        reflection = thoughts.generate_reflection(
            topic=query, prompts=prompts, brain_snap=last_snap,
            stdp_changed_pct=avg_chg, avg_reward=avg_reward,
        )
        print(f"\n  💭 {reflection}", flush=True)

        # 7. 存到海马体
        mem = Memory(
            id=f"mem_{cycle+1:02d}_{int(time.time())}",
            timestamp=time.time(),
            source_url=best_url,
            source_title=best_title,
            topic=query,
            keywords=keywords,
            prompts=prompts,
            reward=avg_reward,
            stdp_changed_pct=avg_chg,
            reflection=reflection,
        )
        hippocampus.store(mem)
        curiosity.record_learning(query, keywords)

        # 记录到 log
        learning_log.append({
            "cycle": cycle + 1,
            "query": query,
            "reason": reason,
            "search_results_count": len(search_results),
            "page_url": best_url,
            "page_title": best_title,
            "page_text_length": len(page['text']),
            "prompts": prompts,
            "keywords": keywords,
            "training_results": [{k: v for k, v in r.items() if k != "snapshots"} for r in cycle_results],
            "avg_reward": avg_reward,
            "avg_stdp_changed_pct": avg_chg,
            "stdp_norm_first": cycle_stdp_first,
            "stdp_norm_last": cycle_stdp_last,
            "last_brain_snap": last_snap,
            "reflection": reflection,
            "thought_state": thoughts.state,
        })

        # gc
        gc.collect()

    total_time = time.time() - t_start
    final_stdp_norm = brain.stdp_weight.norm().item()
    stdp_delta = final_stdp_norm - brain.initial_stdp_norm

    print(f"\n{'=' * 90}", flush=True)
    print(f"✅ 自主学习完成!", flush=True)
    print(f"{'=' * 90}", flush=True)
    print(f"  总耗时: {total_time:.1f}s ({total_time/60:.1f} min)", flush=True)
    print(f"  学习周期: {len(learning_log)}", flush=True)
    print(f"  训练 prompt 总数: {sum(l['training_results'].__len__() for l in learning_log)}", flush=True)
    print(f"  STDP norm: {brain.initial_stdp_norm:.6f} → {final_stdp_norm:.6f} "
          f"(Δ={stdp_delta:+.6f}, {stdp_delta/max(brain.initial_stdp_norm,1e-8)*100:+.3f}%)", flush=True)
    print(f"  海马体记忆: {len(hippocampus.memories)} 条", flush=True)
    print(f"  思维流片段: {len(thoughts.history)} 段", flush=True)

    # 保存结果
    save_results(learning_log, hippocampus, thoughts, brain, total_time, stdp_delta)
    plot_learning(learning_log, brain, stdp_delta)


# ============================================================================
# 10. 保存结果
# ============================================================================
def save_results(learning_log, hippocampus, thoughts, brain, total_time, stdp_delta):
    # JSON
    json_path = os.path.join(DOWNLOAD_DIR, "gpt2_stdpbrain_v4_autolearn.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            "config": {
                "n_cycles": N_LEARNING_CYCLES,
                "prompts_per_cycle": PROMPTS_PER_CYCLE,
                "max_new_tokens": MAX_NEW_TOKENS,
                "stdp_lr": STDP_LR,
                "total_time_s": total_time,
            },
            "initial_stdp_norm": brain.initial_stdp_norm,
            "final_stdp_norm": float(brain.stdp_weight.norm().item()),
            "stdp_delta": stdp_delta,
            "stdp_delta_pct": stdp_delta / max(brain.initial_stdp_norm, 1e-8) * 100,
            "hippocampus_summary": hippocampus.get_summary(),
            "learning_log": learning_log,
            "thoughts_history": thoughts.history,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n💾 JSON: {json_path}", flush=True)

    # TXT report
    txt_path = os.path.join(DOWNLOAD_DIR, "gpt2_stdpbrain_v4_autolearn.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 90 + "\n")
        f.write("GPT-2 × stdpbrain v4 — 自主学习报告\n")
        f.write("=" * 90 + "\n\n")
        f.write(f"Config: {N_LEARNING_CYCLES} cycles × {PROMPTS_PER_CYCLE} prompts × {MAX_NEW_TOKENS} tokens\n")
        f.write(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)\n\n")

        f.write(f"━━━ STDP 突触矩阵 ━━━\n")
        f.write(f"  norm: {brain.initial_stdp_norm:.6f} → {float(brain.stdp_weight.norm().item()):.6f}\n")
        f.write(f"  delta: {stdp_delta:+.6f} ({stdp_delta/max(brain.initial_stdp_norm,1e-8)*100:+.3f}%)\n\n")

        f.write(f"━━━ 海马体记忆 ━━━\n")
        summary = hippocampus.get_summary()
        f.write(f"  总记忆数: {summary['total']}\n")
        f.write(f"  平均 reward: {summary.get('avg_reward', 0):.4f}\n")
        f.write(f"  平均 STDP-change: {summary.get('avg_stdp_changed', 0):.1f}%\n")
        f.write(f"  学到的主题:\n")
        for t in summary.get('topics', []):
            f.write(f"    - {t}\n")
        f.write(f"  所有关键词: {', '.join(summary.get('all_keywords', [])[:20])}\n\n")

        f.write(f"━━━ 各周期详情 ━━━\n")
        for log in learning_log:
            f.write(f"\n[周期 {log['cycle']}]\n")
            f.write(f"  Query: {log['query']!r}\n")
            f.write(f"  Reason: {log['reason']}\n")
            f.write(f"  Source: {log['page_title'][:60]}\n")
            f.write(f"  URL: {log['page_url']}\n")
            f.write(f"  抓取文本: {log['page_text_length']} 字符\n")
            f.write(f"  关键词: {log['keywords']}\n")
            f.write(f"  Prompts:\n")
            for i, p in enumerate(log['prompts']):
                f.write(f"    [{i+1}] {p!r}\n")
            f.write(f"  训练结果:\n")
            for r in log['training_results']:
                f.write(f"    [{r['prompt_idx']+1}] R={r['avg_reward']:+.3f} chg={r['stdp_changed_pct']:.0f}% "
                        f"→ {r['generated_text'][:60]!r}\n")
            f.write(f"  STDP norm: {log['stdp_norm_first']:.4f} → {log['stdp_norm_last']:.4f}\n")
            f.write(f"  Avg reward: {log['avg_reward']:.4f}\n")
            f.write(f"  思维状态: {log['thought_state']}\n")
            f.write(f"  反思: {log['reflection']}\n")

    print(f"💾 TXT: {txt_path}", flush=True)

    # Markdown journal (人类可读的学习日志)
    md_path = os.path.join(DOWNLOAD_DIR, "gpt2_stdpbrain_v4_journal.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# 🧠 STDP Brain 自主学习日志\n\n")
        f.write(f"> brain 系统通过网络自主学习, 用 STDP 三因子规则更新突触, 用海马体存储记忆.\n\n")
        f.write(f"**学习时间**: {total_time:.1f}s ({total_time/60:.1f} min)\n\n")
        f.write(f"**学习规模**: {N_LEARNING_CYCLES} 个主题 × {PROMPTS_PER_CYCLE} 个 prompt × {MAX_NEW_TOKENS} 个 token\n\n")
        f.write(f"**STDP 漂移**: {brain.initial_stdp_norm:.4f} → {float(brain.stdp_weight.norm().item()):.4f} "
                f"({stdp_delta/max(brain.initial_stdp_norm,1e-8)*100:+.3f}%)\n\n")
        f.write("---\n\n")

        for log in learning_log:
            f.write(f"## 📚 周期 {log['cycle']}: {log['query']}\n\n")
            f.write(f"**为什么学这个**: {log['reason']}\n\n")
            f.write(f"**来源**: [{log['page_title']}]({log['page_url']})\n\n")
            f.write(f"**抓取**: {log['page_text_length']} 字符\n\n")
            f.write(f"**关键词**: {', '.join(log['keywords'])}\n\n")
            f.write(f"**学到的内容** (训练 prompt):\n\n")
            for i, p in enumerate(log['prompts']):
                f.write(f"{i+1}. {p}\n")
            f.write(f"\n**STDP 训练结果**:\n\n")
            f.write(f"| # | Reward | STDP改变率 | 生成文本 |\n")
            f.write(f"|---|--------|-----------|----------|\n")
            for r in log['training_results']:
                f.write(f"| {r['prompt_idx']+1} | {r['avg_reward']:+.3f} | {r['stdp_changed_pct']:.0f}% | "
                        f"{r['generated_text'][:50]}... |\n")
            f.write(f"\n**思维状态**: `{log['thought_state']}`\n\n")
            f.write(f"**学习反思**:\n\n")
            f.write(f"> {log['reflection']}\n\n")
            f.write(f"---\n\n")

        f.write("## 🧠 海马体记忆总览\n\n")
        f.write(f"| # | 主题 | 关键词 | Reward | STDP改变率 |\n")
        f.write(f"|---|------|--------|--------|-----------|\n")
        for i, mem in enumerate(hippocampus.memories):
            f.write(f"| {i+1} | {mem.topic} | {', '.join(mem.keywords[:3])} | "
                    f"{mem.reward:+.3f} | {mem.stdp_changed_pct:.0f}% |\n")

    print(f"💾 Journal: {md_path}", flush=True)


# ============================================================================
# 11. 趋势图
# ============================================================================
def plot_learning(learning_log, brain, stdp_delta):
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

    fig, axes = plt.subplots(2, 3, figsize=(20, 10), constrained_layout=True)

    # 1. STDP norm drift across cycles
    ax = axes[0, 0]
    norms_first = [l['stdp_norm_first'] for l in learning_log]
    norms_last = [l['stdp_norm_last'] for l in learning_log]
    cycles = list(range(1, len(learning_log) + 1))
    ax.plot(cycles, norms_first, 'o--', color='tab:blue', label='cycle start', markersize=8)
    ax.plot(cycles, norms_last, 's-', color='tab:red', label='cycle end', markersize=8)
    ax.axhline(brain.initial_stdp_norm, color='gray', linestyle=':', alpha=0.5, label='initial')
    ax.set_title('STDP 突触范数漂移 (按学习周期)', fontsize=12)
    ax.set_xlabel('learning cycle')
    ax.set_ylabel('||STDP_W||')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Reward per cycle
    ax = axes[0, 1]
    rewards = [l['avg_reward'] for l in learning_log]
    ax.bar(cycles, rewards, color='tab:green', alpha=0.7)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_title('每周期平均 reward', fontsize=12)
    ax.set_xlabel('learning cycle')
    ax.set_ylabel('avg reward')
    ax.grid(True, alpha=0.3)

    # 3. STDP change rate per cycle
    ax = axes[0, 2]
    chg_rates = [l['avg_stdp_changed_pct'] for l in learning_log]
    ax.bar(cycles, chg_rates, color='tab:orange', alpha=0.7)
    ax.set_title('每周期 STDP 改变 token 比例', fontsize=12)
    ax.set_xlabel('learning cycle')
    ax.set_ylabel('% tokens changed by STDP')
    ax.grid(True, alpha=0.3)

    # 4. Topics learned
    ax = axes[1, 0]
    topics = [l['query'][:25] for l in learning_log]
    topic_rewards = [l['avg_reward'] for l in learning_log]
    ax.barh(range(len(topics)), topic_rewards, color='tab:purple', alpha=0.7)
    ax.set_yticks(range(len(topics)))
    ax.set_yticklabels(topics, fontsize=9)
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_title('各主题的 reward', fontsize=12)
    ax.set_xlabel('avg reward')
    ax.grid(True, alpha=0.3)

    # 5. DA across cycles
    ax = axes[1, 1]
    da_vals = [l['last_brain_snap'].get('da', 0.5) for l in learning_log]
    ne_vals = [l['last_brain_snap'].get('ne', 0.5) for l in learning_log]
    ax.plot(cycles, da_vals, 'o-', color='tab:purple', label='DA', markersize=8)
    ax.plot(cycles, ne_vals, 's-', color='tab:orange', label='NE', markersize=8)
    ax.set_title('多巴胺 DA / 去甲肾上腺素 NE', fontsize=12)
    ax.set_xlabel('learning cycle')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 6. Valence across cycles
    ax = axes[1, 2]
    val_vals = [l['last_brain_snap'].get('valence', 0) for l in learning_log]
    ax.bar(cycles, val_vals, color=['tab:green' if v > 0 else 'tab:red' for v in val_vals], alpha=0.7)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_title('杏仁核 valence (情绪效价)', fontsize=12)
    ax.set_xlabel('learning cycle')
    ax.set_ylabel('valence')
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"GPT-2 × stdpbrain v4 自主学习 — "
        f"{N_LEARNING_CYCLES} cycles × {PROMPTS_PER_CYCLE} prompts × {MAX_NEW_TOKENS} tokens\n"
        f"STDP norm: {brain.initial_stdp_norm:.4f} → {float(brain.stdp_weight.norm().item()):.4f} "
        f"({stdp_delta/max(brain.initial_stdp_norm,1e-8)*100:+.3f}%)",
        fontsize=13,
    )

    plot_path = os.path.join(DOWNLOAD_DIR, "gpt2_stdpbrain_v4.png")
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"💾 PNG: {plot_path}", flush=True)


if __name__ == "__main__":
    main()
