#!/usr/bin/env python3
"""
brain_daemon.py — 长期运行的类脑学习守护进程
============================================

设计原则:
1. 续传: 每个学习周期后保存 checkpoint (STDP权重/海马体/思维流)
2. 崩溃恢复: 启动时自动加载最新 checkpoint 继续
3. 内存安全: 每周期后 gc + torch cache 清理
4. 信号处理: SIGTERM 优雅退出 (保存状态后再退出)
5. 心跳: 每周期写 heartbeat.json 供 supervisor 监控
6. 日志: 同时写文件 + stdout, 带时间戳

用法:
    python3 brain_daemon.py                    # 默认跑 50 个周期
    python3 brain_daemon.py --max-cycles 100   # 跑 100 个周期
    python3 brain_daemon.py --max-cycles 0     # 无限循环 (由 supervisor 控制)
"""

from __future__ import annotations

import os, sys, json, time, random, subprocess, re, warnings, signal, hashlib, gc, argparse
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
import logging
from logging.handlers import RotatingFileHandler

warnings.filterwarnings("ignore")

# === 路径 ===
STDPBRAIN_DIR = "/home/z/my-project/repos/stdpbrain"
RUNTIME_DIR = "/home/z/my-project/brain_runtime"
DOWNLOAD_DIR = "/home/z/my-project/download"
CKPT_DIR = os.path.join(RUNTIME_DIR, "checkpoints")
LOG_DIR = os.path.join(RUNTIME_DIR, "logs")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

os.chdir(STDPBRAIN_DIR)
sys.path.insert(0, STDPBRAIN_DIR)
sys.path.insert(0, "/home/z/my-project/scripts")

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = os.path.join(RUNTIME_DIR, "cache")
os.environ["HF_TOKEN"] = open(os.path.join(RUNTIME_DIR, ".hf_token")).read().strip()

import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN_SIZE = 768
DEVICE = "cpu"
TORCH_THREADS = 4
torch.set_num_threads(TORCH_THREADS)

# === 学习参数 (每个 session) ===
CYCLES_PER_SESSION = 10        # 每个 daemon 进程跑 10 个周期后退出, supervisor 重启
PROMPTS_PER_CYCLE = 4
MAX_NEW_TOKENS = 8
STDP_LR = 1e-3
STDP_WEIGHT_CLAMP = 0.3

CURIOSITY_TOPICS = [
    "neuroplasticity brain learning", "quantum computing explained",
    "photosynthesis how plants work", "history of artificial intelligence",
    "ocean deep sea creatures", "how vaccines work immune system",
    "black holes spacetime physics", "DNA CRISPR gene editing",
    "climate change carbon cycle", "ancient Egyptian civilization",
    "how memory works hippocampus", "volcanoes plate tectonics",
    "machine learning neural networks", "renaissance art history",
    "symbiosis in nature", "how sleep affects the brain",
    "consciousness philosophy of mind", "evolution natural selection",
    "string theory multiverse", "roman empire fall",
    "how antibodies work", "coral reef ecosystems",
    "quantum entanglement explained", "mayan civilization astronomy",
    "how neurons fire action potential", "plate tectonics earthquakes",
    "industrial revolution history", "stem cells regenerative medicine",
    "greek mythology gods", "how CRISPR edits genes",
    "black hole event horizon", "cognitive behavioral therapy",
    "silk road trade route", "mitochondria cell power",
    "dark matter dark energy", "byzantine empire constantinople",
    "neurotransmitters serotonin dopamine", "great barrier reef",
    "theory of relativity explained", "aztec empire tenochtitlan",
]


# ============================================================================
# 日志配置 (rotating)
# ============================================================================
def setup_logging(session_id: str):
    logger = logging.getLogger("brain_daemon")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    # 文件 handler (rotating 10MB × 5)
    fh = RotatingFileHandler(
        os.path.join(LOG_DIR, f"daemon_{session_id}.log"),
        maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # stdout handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


# ============================================================================
# 复用 v4 的核心组件 (CuriosityEngine, WebFetcher, ContentProcessor, etc.)
# ============================================================================
class CuriosityEngine:
    def __init__(self):
        self.learned_topics: List[str] = []
        self.learned_keywords: List[str] = []
        self.cycle_count = 0

    def generate_query(self) -> Tuple[str, str]:
        self.cycle_count += 1
        if random.random() < 0.3 and self.learned_keywords:
            seed_kw = random.choice(self.learned_keywords)
            associations = {
                "brain": ["consciousness", "neurons", "memory"],
                "neuroplasticity": ["learning", "stroke recovery", "meditation"],
                "quantum": ["entanglement", "uncertainty principle"],
                "DNA": ["RNA", "proteins", "genetics"],
                "climate": ["global warming", "oceans", "weather"],
                "AI": ["machine learning", "neural networks", "deep learning"],
                "memory": ["hippocampus", "sleep", "dreams"],
                "sleep": ["dreams", "circadian rhythm", "REM"],
                "evolution": ["natural selection", "Darwin", "speciation"],
                "black": ["holes", "event horizon", "singularity"],
            }
            related = associations.get(seed_kw, [])
            if related:
                topic = random.choice(related)
                query = f"{seed_kw} {topic}"
                reason = f"[curiosity] 关联探索: 从「{seed_kw}」跳到「{topic}」"
                return query, reason
        topic = random.choice(CURIOSITY_TOPICS)
        reason = f"[curiosity] 第 {self.cycle_count} 周期, 探索「{topic}」"
        return topic, reason

    def record_learning(self, topic: str, keywords: List[str]):
        self.learned_topics.append(topic)
        self.learned_keywords.extend(keywords)

    def state_dict(self):
        return {"learned_topics": self.learned_topics, "learned_keywords": self.learned_keywords, "cycle_count": self.cycle_count}

    def load_state_dict(self, s):
        self.learned_topics = s.get("learned_topics", [])
        self.learned_keywords = s.get("learned_keywords", [])
        self.cycle_count = s.get("cycle_count", 0)


class WebFetcher:
    def __init__(self):
        self.search_cache: Dict[str, List] = {}
        self.page_cache: Dict[str, str] = {}

    def search(self, query: str, num: int = 5) -> List[Dict]:
        if query in self.search_cache:
            return self.search_cache[query]
        cache_file = f"/tmp/brain_search_{hashlib.md5(query.encode()).hexdigest()[:8]}.json"
        try:
            r = subprocess.run(
                ["z-ai", "function", "-n", "web_search",
                 "-a", json.dumps({"query": query, "num": num}),
                 "-o", cache_file],
                capture_output=True, text=True, timeout=30
            )
            if r.returncode != 0: return []
            with open(cache_file) as f: data = json.load(f)
            results = data if isinstance(data, list) else data.get("data", [])
            self.search_cache[query] = results
            return results
        except: return []

    def read_page(self, url: str) -> Optional[Dict]:
        if url in self.page_cache:
            return {"title": "(cached)", "text": self.page_cache[url], "url": url}
        cache_file = f"/tmp/brain_page_{hashlib.md5(url.encode()).hexdigest()[:8]}.json"
        try:
            r = subprocess.run(
                ["z-ai", "function", "-n", "page_reader",
                 "-a", json.dumps({"url": url}), "-o", cache_file],
                capture_output=True, text=True, timeout=60
            )
            if r.returncode != 0: return None
            with open(cache_file) as f: data = json.load(f)
            inner = data.get("data", data)
            title = inner.get("title", "")
            html = inner.get("html", "")
            text = re.sub(r'<script[^>]*>[\s\S]*?</script>', '', html, flags=re.I)
            text = re.sub(r'<style[^>]*>[\s\S]*?</style>', '', text, flags=re.I)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            if len(text) < 100: return None
            self.page_cache[url] = text
            return {"title": title, "text": text, "url": url}
        except: return None


class ContentProcessor:
    @staticmethod
    def extract_prompts(text: str, n: int = PROMPTS_PER_CYCLE) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        candidates = []
        for s in sentences:
            s = s.strip()
            if 30 < len(s) < 200 and re.search(r'[a-zA-Z]{3,}', s):
                if len(s) > 80:
                    cut = s[:80].rfind(' ')
                    if cut > 30: s = s[:cut]
                candidates.append(s)
        seen = set(); unique = []
        for s in candidates:
            if s not in seen:
                seen.add(s); unique.append(s)
        random.shuffle(unique)
        return unique[:n]

    @staticmethod
    def extract_keywords(text: str, topk: int = 5) -> List[str]:
        stop = set("the a an and or but in on at to for of is are was were be been being have has had do does did will would could should may might must can this that these those it its their there here as by with from into out up down over under again further then once".split())
        words = re.findall(r'[a-zA-Z]{4,}', text.lower())
        wf = {}
        for w in words:
            if w not in stop: wf[w] = wf.get(w, 0) + 1
        return [w for w, _ in sorted(wf.items(), key=lambda x: -x[1])[:topk]]


@dataclass
class Memory:
    id: str
    timestamp: float
    source_url: str
    source_title: str
    topic: str
    keywords: List[str]
    prompts: List[str]
    reward: float
    stdp_changed_pct: float
    reflection: str
    cycle: int


class HippocampusLite:
    def __init__(self):
        self.memories: List[Memory] = []

    def store(self, mem: Memory):
        self.memories.append(mem)

    def get_summary(self) -> Dict:
        if not self.memories: return {"total": 0}
        return {
            "total": len(self.memories),
            "topics": [m.topic for m in self.memories],
            "avg_reward": sum(m.reward for m in self.memories) / len(self.memories),
            "avg_stdp_changed": sum(m.stdp_changed_pct for m in self.memories) / len(self.memories),
            "all_keywords": list(set(kw for m in self.memories for kw in m.keywords)),
        }

    def state_dict(self):
        return {"memories": [asdict(m) for m in self.memories]}

    def load_state_dict(self, s):
        self.memories = []
        for m in s.get("memories", []):
            self.memories.append(Memory(**m))


# ============================================================================
# STDP Brain (with checkpoint save/load)
# ============================================================================
class STDPBrain:
    def __init__(self, logger):
        self.logger = logger
        import importlib
        self.modules: Dict[str, nn.Module] = {}
        spec = [
            ("cerebellar", "core.cerebellar_correction_677", "create_cerebellar_correction_system", dict(hidden_size=HIDDEN_SIZE)),
            ("basal_ganglia", "core.basal_ganglia_dopamine", "create_basal_ganglia_dopamine_system", dict(hidden_size=HIDDEN_SIZE)),
            ("lc_ne", "core.locus_coeruleus_ne", "create_locus_coeruleus_ne_system", dict(hidden_size=HIDDEN_SIZE)),
            ("amygdala", "core.amygdala", "create_amygdala_system", dict(hidden_size=HIDDEN_SIZE)),
            ("dual_process", "core.dual_process", "create_dual_process_system", dict(hidden_size=HIDDEN_SIZE)),
            ("synaptic_plast", "core.synaptic_plasticity", "create_synaptic_plasticity_system", dict(hidden_size=HIDDEN_SIZE)),
        ]
        for name, path, factory, kwargs in spec:
            try:
                mod = importlib.import_module(path)
                m = getattr(mod, factory)(**kwargs)
                m.eval() if hasattr(m, "eval") else None
                for p in m.parameters(): p.requires_grad_(False)
                self.modules[name] = m
            except Exception as e:
                logger.warning(f"  {name}: {e}")

        self.stdp_weight = nn.Parameter(
            torch.randn(HIDDEN_SIZE, HIDDEN_SIZE, device=DEVICE) * 0.01, requires_grad=True
        )
        self.optimizer = torch.optim.SGD([self.stdp_weight], lr=STDP_LR, momentum=0.9, weight_decay=1e-4)

        sp = self.modules.get("synaptic_plast")
        self.sp_meta_params: Dict[str, Any] = {}
        if sp is not None:
            for sub_name, sub_mod in sp.named_modules():
                for attr in ['a_plus', 'a_minus']:
                    if hasattr(sub_mod, attr):
                        self.sp_meta_params[f"{sub_name}.{attr}"] = getattr(sub_mod, attr)

        self.da_level = 0.5; self.ne_level = 0.5; self.valence = 0.0
        self.initial_stdp_norm = float(self.stdp_weight.norm().item())

    def step(self, hidden_now, hidden_prev, external_reward):
        snap = {}
        H = hidden_now.detach()
        with torch.no_grad():
            stdp_delta = F.linear(H, self.stdp_weight)
            modified_hidden = H + stdp_delta * 0.1

            cb = self.modules.get("cerebellar")
            cerebellar_err = 0.0; corrected = H
            if cb is not None:
                try:
                    r = cb.forward(H, hidden_prev, H)
                    err = r.get("prediction_error", None)
                    if isinstance(err, torch.Tensor): cerebellar_err = float(err.norm().item())
                    corrected = r.get("corrected_output", H)
                    if not isinstance(corrected, torch.Tensor): corrected = H
                except: pass
            snap["cerebellar_error"] = cerebellar_err

            bg = self.modules.get("basal_ganglia")
            da_level = 0.5; bg_out = corrected
            if bg is not None:
                try:
                    r = bg.forward(corrected)
                    if hasattr(bg, "get_dopamine_level"):
                        da = bg.get_dopamine_level()
                        da_level = float(da.mean().item()) if isinstance(da, torch.Tensor) else float(da)
                    bg_out = r.get("output", corrected)
                    if not isinstance(bg_out, torch.Tensor): bg_out = corrected
                except: pass
            snap["da"] = da_level; self.da_level = da_level

            lc = self.modules.get("lc_ne")
            ne_level = 0.5; lc_out = bg_out
            if lc is not None:
                try:
                    r = lc.forward(bg_out)
                    if hasattr(lc, "get_ne_level"):
                        ne = lc.get_ne_level()
                        ne_level = float(ne.mean().item()) if isinstance(ne, torch.Tensor) else float(ne)
                    lc_out = r.get("output", bg_out)
                    if not isinstance(lc_out, torch.Tensor): lc_out = bg_out
                except: pass
            snap["ne"] = ne_level; self.ne_level = ne_level

            amy = self.modules.get("amygdala")
            valence = 0.0; amy_out = lc_out
            if amy is not None:
                try:
                    amy_in = lc_out if lc_out.dim() == 2 else lc_out.unsqueeze(0)
                    r = amy.forward(amy_in)
                    v = r.get("stats", {}).get("valence", 0)
                    if isinstance(v, torch.Tensor): v = float(v.mean().item())
                    valence = float(v)
                    amy_out = r.get("output", amy_in)
                    if not isinstance(amy_out, torch.Tensor): amy_out = amy_in
                except: pass
            snap["valence"] = valence; self.valence = valence

            dp = self.modules.get("dual_process")
            system_mode = "?"
            if dp is not None:
                try:
                    r = dp.forward(amy_out)
                    system_mode = str(r.get("stats", {}).get("active_system", "?"))
                except: pass
            snap["system"] = system_mode

            sp = self.modules.get("synaptic_plast")
            delta_w = None
            if sp is not None:
                try:
                    new_w = sp.forward(pre_activity=hidden_prev, post_activity=H,
                                       dopamine_level=da_level, weights=self.stdp_weight.data.clone())
                    delta_w = (new_w - self.stdp_weight.data).detach()
                except: delta_w = None

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
                meta_lr = 0.0005
                for name, param in self.sp_meta_params.items():
                    if "a_plus" in name.lower():
                        if isinstance(param, torch.Tensor):
                            param.add_(combined_reward * meta_lr); param.clamp_(0.0001, 0.1)
                        else:
                            nv = float(param) + combined_reward * meta_lr
                            nv = max(0.0001, min(0.1, nv))
                            parts = name.split("."); obj = self.modules["synaptic_plast"]
                            for p in parts[:-1]:
                                if p: obj = getattr(obj, p)
                            setattr(obj, parts[-1], nv)
                            self.sp_meta_params[name] = nv
                    elif "a_minus" in name.lower():
                        if isinstance(param, torch.Tensor):
                            param.sub_(combined_reward * meta_lr); param.clamp_(-0.1, -0.0001)
                        else:
                            nv = float(param) - combined_reward * meta_lr
                            nv = max(-0.1, min(-0.0001, nv))
                            parts = name.split("."); obj = self.modules["synaptic_plast"]
                            for p in parts[:-1]:
                                if p: obj = getattr(obj, p)
                            setattr(obj, parts[-1], nv)
                            self.sp_meta_params[name] = nv

        snap["stdp_norm"] = float(self.stdp_weight.norm().item())
        snap["stdp_grad_norm"] = float(self.stdp_weight.grad.norm().item()) if self.stdp_weight.grad is not None else 0.0

        if self.stdp_weight.grad is not None:
            gn = self.stdp_weight.grad.norm().item()
            if gn > 10.0: self.stdp_weight.grad.mul_(10.0 / gn)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.stdp_weight.data.clamp_(-STDP_WEIGHT_CLAMP, STDP_WEIGHT_CLAMP)
        return snap, modified_hidden.detach()

    def save_checkpoint(self, path: str):
        """保存 STDP 权重 + optimizer 状态 + sp 元参数."""
        sp_meta_serializable = {}
        for k, v in self.sp_meta_params.items():
            sp_meta_serializable[k] = float(v) if not isinstance(v, torch.Tensor) else float(v.mean().item())
        ckpt = {
            "stdp_weight": self.stdp_weight.data.clone(),
            "optimizer_state": self.optimizer.state_dict(),
            "sp_meta_params": sp_meta_serializable,
            "initial_stdp_norm": self.initial_stdp_norm,
            "da_level": self.da_level, "ne_level": self.ne_level, "valence": self.valence,
        }
        torch.save(ckpt, path)

    def load_checkpoint(self, path: str):
        if not os.path.exists(path): return False
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        self.stdp_weight.data.copy_(ckpt["stdp_weight"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        # 恢复 sp 元参数
        for name, val in ckpt.get("sp_meta_params", {}).items():
            if name in self.sp_meta_params:
                parts = name.split("."); obj = self.modules["synaptic_plast"]
                for p in parts[:-1]:
                    if p: obj = getattr(obj, p)
                setattr(obj, parts[-1], val)
                self.sp_meta_params[name] = val
        self.initial_stdp_norm = ckpt.get("initial_stdp_norm", self.initial_stdp_norm)
        self.da_level = ckpt.get("da_level", 0.5)
        self.ne_level = ckpt.get("ne_level", 0.5)
        self.valence = ckpt.get("valence", 0.0)
        return True


# ============================================================================
# 训练逻辑
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
        else: stdp_bonus = 0.0
        log_probs = torch.log(probs + 1e-8)
        entropy = -float((probs * log_probs).sum().item())
        entropy_bonus = 0.05 * min(entropy, 5.0) / 5.0
    return 0.4 * confidence + 0.3 * rep_penalty + 0.2 * stdp_bonus + 0.1 * entropy_bonus


def train_one_prompt(tokenizer, gpt2, brain, prompt, max_new_tokens, logger):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    generated_ids = input_ids.clone()
    with torch.no_grad():
        out0 = gpt2(input_ids, output_hidden_states=True, use_cache=False)
        hidden_prev = out0.hidden_states[-1][:, -1, :].squeeze(0).detach()
        if hidden_prev.dim() == 1: hidden_prev = hidden_prev.unsqueeze(0)

    snapshots = []; rewards = []; stdp_changed_count = 0
    for step in range(max_new_tokens):
        with torch.no_grad():
            out = gpt2(generated_ids, output_hidden_states=True, use_cache=False)
            hidden_now = out.hidden_states[-1][:, -1, :].squeeze(0).detach()
            if hidden_now.dim() == 1: hidden_now = hidden_now.unsqueeze(0)
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
        snap["stdp_changed"] = (next_id != base_id)
        if snap["stdp_changed"]: stdp_changed_count += 1
        snapshots.append(snap)
        generated_ids = torch.cat([generated_ids, torch.tensor([[next_id]], device=DEVICE)], dim=1)
        hidden_prev = hidden_now.clone()
        if next_id == tokenizer.eos_token_id: break

    return {
        "prompt": prompt,
        "generated_text": prompt + "".join(s["token"] for s in snapshots),
        "avg_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "stdp_changed_pct": stdp_changed_count / len(snapshots) * 100 if snapshots else 0,
        "n_tokens": len(snapshots),
        "last_brain_snap": snapshots[-1] if snapshots else {},
    }


# ============================================================================
# Checkpoint 管理
# ============================================================================
def save_full_checkpoint(brain, hippocampus, curiosity, cycle_num, session_id, logger):
    """保存完整状态: brain + hippocampus + curiosity + 元数据."""
    ckpt_path = os.path.join(CKPT_DIR, f"brain_ckpt_{session_id}_c{cycle_num:04d}.pt")
    brain.save_checkpoint(ckpt_path)

    state_path = os.path.join(CKPT_DIR, f"brain_state_{session_id}_c{cycle_num:04d}.json")
    state = {
        "cycle": cycle_num,
        "session_id": session_id,
        "timestamp": time.time(),
        "hippocampus": hippocampus.state_dict(),
        "curiosity": curiosity.state_dict(),
        "stdp_norm": float(brain.stdp_weight.norm().item()),
        "initial_stdp_norm": brain.initial_stdp_norm,
    }
    with open(state_path, 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=2, default=str)

    # 更新 latest 指针
    latest_path = os.path.join(CKPT_DIR, "LATEST")
    with open(latest_path, 'w') as f:
        f.write(f"{session_id}:{cycle_num:04d}\n")

    logger.info(f"  💾 checkpoint saved: cycle={cycle_num} stdp_norm={state['stdp_norm']:.6f}")


def load_latest_checkpoint(brain, hippocampus, curiosity, logger):
    """加载最新的 checkpoint. 返回 (cycle_num, session_id) 或 (0, new_session_id)."""
    latest_path = os.path.join(CKPT_DIR, "LATEST")
    if not os.path.exists(latest_path):
        logger.info("  [*] no previous checkpoint, starting fresh")
        return 0, None

    with open(latest_path) as f:
        line = f.read().strip()
    parts = line.split(":")
    if len(parts) != 2:
        logger.info("  [*] invalid LATEST file, starting fresh")
        return 0, None
    prev_session, prev_cycle_str = parts
    prev_cycle = int(prev_cycle_str)

    ckpt_path = os.path.join(CKPT_DIR, f"brain_ckpt_{prev_session}_c{prev_cycle:04d}.pt")
    state_path = os.path.join(CKPT_DIR, f"brain_state_{prev_session}_c{prev_cycle:04d}.json")

    if not os.path.exists(ckpt_path) or not os.path.exists(state_path):
        logger.info(f"  [*] checkpoint files missing, starting fresh")
        return 0, None

    try:
        brain.load_checkpoint(ckpt_path)
        with open(state_path) as f:
            state = json.load(f)
        hippocampus.load_state_dict(state.get("hippocampus", {}))
        curiosity.load_state_dict(state.get("curiosity", {}))
        logger.info(f"  ✅ resumed from session={prev_session} cycle={prev_cycle} "
                    f"(stdp_norm={state.get('stdp_norm', 0):.6f}, memories={len(hippocampus.memories)})")
        return prev_cycle, prev_session
    except Exception as e:
        logger.warning(f"  [!] failed to load checkpoint: {e}, starting fresh")
        return 0, None


# ============================================================================
# 心跳
# ============================================================================
def write_heartbeat(session_id, cycle, total_cycles, brain, hippocampus, status="running"):
    hb = {
        "session_id": session_id,
        "cycle": cycle,
        "total_cycles_this_session": total_cycles,
        "status": status,
        "timestamp": time.time(),
        "stdp_norm": float(brain.stdp_weight.norm().item()),
        "initial_stdp_norm": brain.initial_stdp_norm,
        "stdp_delta_pct": (float(brain.stdp_weight.norm().item()) - brain.initial_stdp_norm) / max(brain.initial_stdp_norm, 1e-8) * 100,
        "memories": len(hippocampus.memories),
        "da": brain.da_level, "ne": brain.ne_level, "valence": brain.valence,
        "pid": os.getpid(),
    }
    hb_path = os.path.join(RUNTIME_DIR, "heartbeat.json")
    with open(hb_path, 'w') as f:
        json.dump(hb, f, indent=2)


# ============================================================================
# 信号处理
# ============================================================================
class GracefulExit:
    def __init__(self):
        self.should_exit = False
        signal.signal(signal.SIGTERM, self._handler)
        signal.signal(signal.SIGINT, self._handler)

    def _handler(self, signum, frame):
        logging.getLogger("brain_daemon").info(f"  [signal] received signal {signum}, will exit after current cycle")
        self.should_exit = True


# ============================================================================
# 主循环
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cycles", type=int, default=CYCLES_PER_SESSION,
                        help="max cycles this session (0=infinite)")
    args = parser.parse_args()

    session_id = time.strftime("%Y%m%d_%H%M%S")
    logger = setup_logging(session_id)

    logger.info("=" * 80)
    logger.info("🧠 brain_daemon starting")
    logger.info("=" * 80)
    logger.info(f"  session_id: {session_id}")
    logger.info(f"  max_cycles: {args.max_cycles if args.max_cycles > 0 else 'infinite'}")
    logger.info(f"  device: {DEVICE}, threads: {TORCH_THREADS}")
    logger.info(f"  HF_TOKEN: configured (from {RUNTIME_DIR}/.hf_token)")

    # 加载 GPT-2
    logger.info("  loading GPT-2...")
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt2.eval(); gpt2.to(DEVICE)
    for p in gpt2.parameters(): p.requires_grad_(False)
    logger.info(f"  ✅ GPT-2 loaded ({sum(p.numel() for p in gpt2.parameters())/1e6:.1f}M)")

    # 初始化 brain + 子系统
    brain = STDPBrain(logger)
    curiosity = CuriosityEngine()
    fetcher = WebFetcher()
    hippocampus = HippocampusLite()

    # 加载 checkpoint
    prev_cycle, prev_session = load_latest_checkpoint(brain, hippocampus, curiosity, logger)
    start_cycle = prev_cycle
    global_cycle = prev_cycle

    # 信号处理
    graceful = GracefulExit()

    logger.info(f"  starting at global_cycle={global_cycle}, initial_stdp_norm={brain.initial_stdp_norm:.6f}")
    logger.info("=" * 80)

    t_session_start = time.time()
    cycles_this_session = 0

    while not graceful.should_exit:
        if args.max_cycles > 0 and cycles_this_session >= args.max_cycles:
            logger.info(f"  reached max_cycles={args.max_cycles}, exiting session cleanly")
            break

        global_cycle += 1
        cycles_this_session += 1
        logger.info(f"\n{'─' * 80}")
        logger.info(f"📚 cycle {global_cycle} (session cycle {cycles_this_session})")
        logger.info(f"{'─' * 80}")

        try:
            # 1. 好奇心
            query, reason = curiosity.generate_query()
            logger.info(f"  💭 {reason}")
            logger.info(f"  🔍 query: {query!r}")

            # 2. 搜索
            t0 = time.time()
            results = fetcher.search(query, num=5)
            logger.info(f"  📋 found {len(results)} results ({time.time()-t0:.1f}s)")
            if not results:
                logger.warning("  ⚠️ no results, skipping")
                continue

            # 3. 选最佳 URL
            best_url = None; best_title = ""
            for r in results:
                host = r.get('host_name', '')
                if any(d in host for d in ['wikipedia', 'britannica', 'nature', 'sciencedaily', 'ncbi', 'nasa']):
                    best_url = r.get('url'); best_title = r.get('name', ''); break
            if not best_url:
                best_url = results[0].get('url'); best_title = results[0].get('name', '')

            logger.info(f"  📖 reading: {best_title[:60]}")

            # 4. 抓取
            t0 = time.time()
            page = fetcher.read_page(best_url)
            if not page:
                logger.warning("  ⚠️ page read failed, skipping")
                continue
            logger.info(f"  📄 fetched {len(page['text'])} chars ({time.time()-t0:.1f}s)")

            # 5. 切片
            prompts = ContentProcessor.extract_prompts(page['text'], n=PROMPTS_PER_CYCLE)
            keywords = ContentProcessor.extract_keywords(page['text'], topk=5)
            logger.info(f"  ✂️ {len(prompts)} prompts, keywords: {keywords}")

            # 6. STDP 训练
            cycle_stdp_first = brain.stdp_weight.norm().item()
            cycle_results = []
            for i, prompt in enumerate(prompts):
                result = train_one_prompt(tokenizer, gpt2, brain, prompt, MAX_NEW_TOKENS, logger)
                result["prompt_idx"] = i
                cycle_results.append(result)
                logger.info(f"     [{i+1}] R={result['avg_reward']:+.3f} chg={result['stdp_changed_pct']:.0f}% | "
                            f"→ {result['generated_text'][:50]!r}")

            cycle_stdp_last = brain.stdp_weight.norm().item()
            avg_reward = sum(r["avg_reward"] for r in cycle_results) / len(cycle_results)
            avg_chg = sum(r["stdp_changed_pct"] for r in cycle_results) / len(cycle_results)
            last_snap = cycle_results[-1]["last_brain_snap"]

            logger.info(f"  📊 STDP: {cycle_stdp_first:.6f} → {cycle_stdp_last:.6f} "
                        f"({(cycle_stdp_last-cycle_stdp_first)/max(cycle_stdp_first,1e-8)*100:+.4f}%)")
            logger.info(f"  📊 avg_reward={avg_reward:.4f}, avg_chg={avg_chg:.1f}%")

            # 7. 思维流反思
            thought_state = "FOCUSED" if brain.da_level > 0.55 else "RESTING" if brain.ne_level < 0.3 else "REFLECTING" if brain.valence < -0.05 else "WANDERING"
            if avg_chg > 40:
                reflection_note = "STDP 大量改变输出——新知识在重塑我的思考"
            elif avg_chg > 20:
                reflection_note = "STDP 部分改变输出——我在吸收新知识"
            else:
                reflection_note = "STDP 改变较少——这个领域可能已熟悉"
            reflection = (f"[{thought_state}] 学了「{query}」, {len(prompts)} 个 prompt, "
                         f"R={avg_reward:.3f}, chg={avg_chg:.0f}%. {reflection_note}. "
                         f"DA={brain.da_level:.3f}, NE={brain.ne_level:.3f}, val={brain.valence:+.3f}")
            logger.info(f"  💭 {reflection}")

            # 8. 存海马体
            mem = Memory(
                id=f"mem_{global_cycle:04d}_{int(time.time())}",
                timestamp=time.time(), source_url=best_url, source_title=best_title,
                topic=query, keywords=keywords, prompts=prompts,
                reward=avg_reward, stdp_changed_pct=avg_chg,
                reflection=reflection, cycle=global_cycle,
            )
            hippocampus.store(mem)
            curiosity.record_learning(query, keywords)

            # 9. 保存 checkpoint
            save_full_checkpoint(brain, hippocampus, curiosity, global_cycle, session_id, logger)

            # 10. 心跳
            write_heartbeat(session_id, global_cycle, cycles_this_session, brain, hippocampus)

            # 11. 内存清理
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"  ❌ cycle {global_cycle} failed: {e}", exc_info=True)
            # 仍然保存 checkpoint (状态可能部分更新)
            try:
                save_full_checkpoint(brain, hippocampus, curiosity, global_cycle, session_id, logger)
            except: pass
            # 短暂休眠避免快速失败循环
            time.sleep(5)

    # 优雅退出
    total_time = time.time() - t_session_start
    final_norm = float(brain.stdp_weight.norm().item())
    logger.info("\n" + "=" * 80)
    logger.info("🧠 session complete")
    logger.info("=" * 80)
    logger.info(f"  session_id: {session_id}")
    logger.info(f"  cycles this session: {cycles_this_session}")
    logger.info(f"  total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info(f"  STDP norm: {brain.initial_stdp_norm:.6f} → {final_norm:.6f} "
                f"({(final_norm-brain.initial_stdp_norm)/max(brain.initial_stdp_norm,1e-8)*100:+.4f}%)")
    logger.info(f"  hippocampus memories: {len(hippocampus.memories)}")
    logger.info(f"  checkpoint saved, supervisor can restart to continue")

    write_heartbeat(session_id, global_cycle, cycles_this_session, brain, hippocampus, status="exited_clean")


if __name__ == "__main__":
    main()
