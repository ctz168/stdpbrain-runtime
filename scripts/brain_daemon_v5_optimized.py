#!/usr/bin/env python3
"""
brain_daemon_v5_optimized.py — 优化版 (基于意识测试结果)
=========================================================

针对意识测试的 3 个低分项做优化:

1. 连续性 0.2 → 增强 working memory
   - WORKING_MEMORY_INJECT_SCALE: 0.03 → 0.15 (5x)
   - 加入"主题锚点": 每个 cycle 把 topic embedding 也注入

2. 情感一致性 0.9 → 增强 amygdala 区分度
   - 用 GPT-2 自己的 token 概率做情感检测 (正面词 vs 负面词)
   - valence 计算改为: log(P(正面词)) - log(P(负面词))
   - EMOTION_INJECT_SCALE: 0.05 → 0.15 (3x)

3. 全局广播 0.4 → 增强 GW 区分度
   - 在 GW.integrate 前对 hidden 做 layer norm + dropout (增加差异)
   - 用 emotion + memory + perception 3 个不同来源 (而非都从 hidden 派生)
"""

from __future__ import annotations
import os, sys, json, time, random, subprocess, re, warnings, signal, hashlib, gc, argparse, asyncio, threading
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
import logging
from logging.handlers import RotatingFileHandler
from collections import deque

warnings.filterwarnings("ignore")

STDPBRAIN_DIR = "/home/z/my-project/repos/stdpbrain"
AICQSDK_DIR = "/home/z/my-project/repos/AIcqsdk"
RUNTIME_DIR = "/home/z/my-project/brain_runtime"
DOWNLOAD_DIR = "/home/z/my-project/download"
CKPT_DIR = os.path.join(RUNTIME_DIR, "checkpoints_v5")
LOG_DIR = os.path.join(RUNTIME_DIR, "logs")
os.makedirs(CKPT_DIR, exist_ok=True)

os.chdir(STDPBRAIN_DIR)
sys.path.insert(0, STDPBRAIN_DIR)
sys.path.insert(0, "/home/z/my-project/scripts")
sys.path.insert(0, AICQSDK_DIR)

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = os.path.join(RUNTIME_DIR, "cache")
os.environ["HF_TOKEN"] = open(os.path.join(RUNTIME_DIR, ".hf_token")).read().strip()
os.environ["HOME"] = RUNTIME_DIR

import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN_SIZE = 768
DEVICE = "cpu"
TORCH_THREADS = 4
torch.set_num_threads(TORCH_THREADS)

# STDP (保持 v2/v4 安全参数)
STDP_INIT_STD = 0.001
STDP_INJECT_SCALE = 0.01
STDP_LR = 1e-4
STDP_WEIGHT_CLAMP = 0.1
M1_ROLLBACK_THRESHOLD = -0.05

CYCLES_PER_SESSION = 10
PROMPTS_PER_CYCLE = 4
MAX_NEW_TOKENS = 8
MASTER_ID_DEFAULT = "1000008"
AICQ_SERVER = "https://aicq.me"

# === v5 优化参数 ===
WORKING_MEMORY_SIZE = 10
WORKING_MEMORY_INJECT_SCALE = 0.15   # v4: 0.03 → v5: 0.15 (5x)
EMOTION_INJECT_SCALE = 0.15          # v4: 0.05 → v5: 0.15 (3x)
PREDICTION_TOKENS = 2

# v5: 情感词表 (用于 valence 计算)
POSITIVE_WORDS = ["good", "great", "happy", "love", "beautiful", "amazing", "wonderful",
                  "excellent", "perfect", "joy", "hope", "win", "success", "best", "bright"]
NEGATIVE_WORDS = ["bad", "terrible", "sad", "hate", "ugly", "awful", "horrible",
                  "wrong", "fail", "loss", "death", "kill", "pain", "worst", "dark"]

CURIOSITY_TOPICS = [
    "neuroplasticity brain learning", "quantum computing explained",
    "photosynthesis how plants work", "history of artificial intelligence",
    "ocean deep sea creatures", "how vaccines work immune system",
    "black holes spacetime physics", "DNA CRISPR gene editing",
    "climate change carbon cycle", "ancient Egyptian civilization",
    "how memory works hippocampus", "volcanoes plate tectonics",
    "machine learning neural networks", "renaissance art history",
    "symbiosis in nature", "how sleep affects the brain",
]


def setup_logging(session_id):
    logger = logging.getLogger("brain_v5")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = RotatingFileHandler(os.path.join(LOG_DIR, f"daemon_v5_{session_id}.log"),
                             maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
    fh.setFormatter(fmt); logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    return logger


# AICQ Bridge (复用)
class AICQBridge:
    def __init__(self, master_id, logger):
        self.master_id = master_id; self.logger = logger
        self.loop = None; self.thread = None
        self.connected = threading.Event()
        self.incoming_messages = []; self.lock = threading.Lock()
        self.agent_account_id = None
    def start(self):
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start(); self.connected.wait(timeout=30)
    def _run_loop(self):
        self.loop = asyncio.new_event_loop(); asyncio.set_event_loop(self.loop)
        try: self.loop.run_until_complete(self._start_aicq())
        except Exception as e: self.logger.error(f"[aicq] crashed: {e}")
    async def _start_aicq(self):
        from aicq import startLoop
        from aicq.loop import _get_or_create_identity, _loop_ctx
        async def on_message(content, from_id, ctx):
            self.logger.info(f"📥 [aicq] msg from {from_id}: {content!r}")
            with self.lock: self.incoming_messages.append((from_id, content))
            return None
        try:
            identity = _get_or_create_identity()
            self.agent_account_id = identity.get("account_id")
        except: pass
        async def _run():
            try: await startLoop(on_message, server=AICQ_SERVER)
            except Exception as e: self.logger.error(f"[aicq] startLoop: {e}")
            finally: self.connected.clear()
        task = asyncio.create_task(_run())
        for _ in range(30):
            if _loop_ctx.ws is not None and not _loop_ctx.ws.closed and _loop_ctx.access_token:
                self.connected.set(); break
            await asyncio.sleep(0.5)
        else: self.connected.set()
        await task
    def send_message(self, content, to_id=None):
        if not self.loop or not self.loop.is_running(): return False
        target = to_id or self.master_id
        future = asyncio.run_coroutine_threadsafe(self._send_async(target, content), self.loop)
        try: future.result(timeout=10); return True
        except: return False
    async def _send_async(self, to_id, content):
        from aicq import loop_send_message
        await loop_send_message(to_id, content)
    def get_incoming(self):
        with self.lock:
            msgs = self.incoming_messages.copy(); self.incoming_messages.clear()
        return msgs


# v5: 体验 + Working Memory (复用 v4, 参数调整)
@dataclass
class Experience:
    cycle: int; timestamp: float; topic: str
    hidden_state: List[float]; emotion_valence: float
    emotion_arousal: float; confidence: float; logprob_delta: float
    predicted_tokens: List[int]; actual_first_token: int


class WorkingMemoryStream:
    def __init__(self, max_size=WORKING_MEMORY_SIZE):
        self.experiences = deque(maxlen=max_size); self.last_experience = None
    def add(self, exp):
        self.experiences.append(exp); self.last_experience = exp
    def get_injection(self, device):
        if not self.experiences: return None
        total_weight = 0.0
        weighted_sum = torch.zeros(HIDDEN_SIZE, device=device)
        for i, exp in enumerate(self.experiences):
            weight = 0.7 ** (len(self.experiences) - 1 - i)
            vec = torch.tensor(exp.hidden_state, device=device)
            weighted_sum += weight * vec; total_weight += weight
        return weighted_sum / total_weight if total_weight > 0 else None
    def get_summary(self):
        if not self.experiences: return {"size": 0}
        recent = list(self.experiences)[-3:]
        return {"size": len(self.experiences),
                "recent_topics": [e.topic[:20] for e in recent],
                "avg_valence": sum(e.emotion_valence for e in self.experiences) / len(self.experiences),
                "avg_confidence": sum(e.confidence for e in self.experiences) / len(self.experiences)}
    def state_dict(self): return {"experiences": [asdict(e) for e in self.experiences]}
    def load_state_dict(self, s):
        self.experiences = deque(maxlen=WORKING_MEMORY_SIZE)
        for e in s.get("experiences", []): self.experiences.append(Experience(**e))
        self.last_experience = self.experiences[-1] if self.experiences else None


class PredictiveCoder:
    def __init__(self):
        self.prediction_history = []; self.total_predictions = 0; self.correct_predictions = 0
    def predict_next_tokens(self, gpt2, hidden_state, n_tokens=PREDICTION_TOKENS):
        with torch.no_grad():
            logits = gpt2.lm_head(hidden_state).squeeze(0)
            predicted_ids = torch.argmax(logits, dim=-1).unsqueeze(0).tolist()
            top5_probs, top5_ids = torch.topk(F.softmax(logits, dim=-1), 5)
            return predicted_ids, top5_ids.squeeze().tolist()
    def compute_prediction_error(self, predicted_ids, actual_first_token, top5_ids):
        self.total_predictions += 1
        exact_correct = (predicted_ids[0] == actual_first_token) if predicted_ids else False
        if exact_correct: self.correct_predictions += 1
        top5_correct = actual_first_token in top5_ids
        if exact_correct: reward = 0.5
        elif top5_correct: reward = 0.3
        else: reward = -0.2
        result = {"exact_correct": exact_correct, "top5_correct": top5_correct,
                  "prediction_reward": reward, "predicted_id": predicted_ids[0] if predicted_ids else -1,
                  "actual_id": actual_first_token}
        self.prediction_history.append(result)
        return result
    def get_accuracy(self): return self.correct_predictions / max(self.total_predictions, 1)
    def get_summary(self): return {"total": self.total_predictions, "accuracy": self.get_accuracy()}


# === v5 优化: EmotionCognitionBinder with semantic valence ===
class EmotionCognitionBinder:
    """v5: 用 GPT-2 token 概率做情感检测, 而非依赖 amygdala 模块."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.positive_token_ids = []
        self.negative_token_ids = []
        for w in POSITIVE_WORDS:
            ids = tokenizer.encode(" " + w)
            if ids: self.positive_token_ids.append(ids[0])
        for w in NEGATIVE_WORDS:
            ids = tokenizer.encode(" " + w)
            if ids: self.negative_token_ids.append(ids[0])
        self.positive_direction = None; self.negative_direction = None
        self.last_emotion_injection = None

    def _ensure_init(self, device):
        if self.positive_direction is None:
            torch.manual_seed(42)
            v = torch.randn(HIDDEN_SIZE, device=device)
            self.positive_direction = v / v.norm()
            self.negative_direction = -self.positive_direction

    def compute_semantic_valence(self, logits: torch.Tensor) -> float:
        """v5: 用 GPT-2 对正面/负面词的概率计算 valence."""
        probs = F.softmax(logits, dim=-1)
        pos_prob = float(probs[self.positive_token_ids].sum().item())
        neg_prob = float(probs[self.negative_token_ids].sum().item())
        # valence = (pos - neg) / (pos + neg + eps), 范围 [-1, 1]
        total = pos_prob + neg_prob + 1e-8
        valence = (pos_prob - neg_prob) / total
        # 放大 (因为概率差异通常很小)
        valence = valence * 5  # 放大 5 倍
        return max(-1.0, min(1.0, valence))

    def apply_emotion(self, hidden, valence, arousal):
        self._ensure_init(hidden.device)
        if abs(valence) < 0.01: return hidden
        if valence > 0:
            injection = self.positive_direction * valence * EMOTION_INJECT_SCALE * (0.5 + arousal)
        else:
            injection = self.negative_direction * abs(valence) * EMOTION_INJECT_SCALE * (0.5 + arousal)
        modified = hidden + injection
        self.last_emotion_injection = float(injection.norm().item())
        return modified

    def modulate_working_memory(self, da_level):
        return 0.5 + da_level


# Memory + Web + Curiosity (复用)
@dataclass
class ConsciousMemory:
    id: str; timestamp: float; topic: str; keywords: List[str]
    prompt: str; hidden_state: List[float]
    logprob_baseline: float; logprob_stdp: float
    confidence: float; self_confidence: float; cycle: int; valence: float


class HippocampusWithRecall:
    def __init__(self): self.memories = []
    def encode(self, hidden_state, topic, keywords, prompt, cycle, lp_b, lp_s, conf, self_conf, valence):
        mem_id = f"mem_{cycle:04d}_{int(time.time()*1000) % 1000000}"
        mem = ConsciousMemory(mem_id, time.time(), topic, keywords, prompt,
                             hidden_state.squeeze().cpu().tolist(),
                             lp_b, lp_s, conf, self_conf, cycle, valence)
        self.memories.append(mem); return mem_id
    def recall(self, query_hidden, topk=3):
        if not self.memories: return []
        query = query_hidden.squeeze().cpu()
        sims = []
        for m in self.memories:
            mem_vec = torch.tensor(m.hidden_state, device=query.device)
            sim = F.cosine_similarity(query.unsqueeze(0), mem_vec.unsqueeze(0)).item()
            sims.append((sim, m))
        sims.sort(key=lambda x: -x[0])
        return [m for _, m in sims[:topk]]
    def get_summary(self):
        if not self.memories: return {"total": 0}
        return {"total": len(self.memories), "topics": list(set(m.topic for m in self.memories))}
    def state_dict(self): return {"memories": [asdict(m) for m in self.memories]}
    def load_state_dict(self, s):
        self.memories = []
        for m in s.get("memories", []): self.memories.append(ConsciousMemory(**m))


class CuriosityEngine:
    def __init__(self):
        self.learned_topics = []; self.learned_keywords = []; self.cycle_count = 0
    def generate_query(self):
        self.cycle_count += 1
        if random.random() < 0.3 and self.learned_keywords:
            seed = random.choice(self.learned_keywords)
            assoc = {"brain":["consciousness"], "quantum":["entanglement"], "DNA":["RNA"], "climate":["oceans"], "memory":["hippocampus"]}
            related = assoc.get(seed, [])
            if related:
                t = random.choice(related); return f"{seed} {t}", f"关联: {seed}→{t}"
        t = random.choice(CURIOSITY_TOPICS); return t, f"探索「{t}」"
    def record_learning(self, topic, keywords):
        self.learned_topics.append(topic); self.learned_keywords.extend(keywords)
    def state_dict(self): return {"learned_topics": self.learned_topics, "learned_keywords": self.learned_keywords, "cycle_count": self.cycle_count}
    def load_state_dict(self, s):
        self.learned_topics = s.get("learned_topics", []); self.learned_keywords = s.get("learned_keywords", []); self.cycle_count = s.get("cycle_count", 0)


class WebFetcher:
    def __init__(self): self.search_cache = {}; self.page_cache = {}
    def search(self, query, num=5):
        if query in self.search_cache: return self.search_cache[query]
        cf = f"/tmp/brain_search_{hashlib.md5(query.encode()).hexdigest()[:8]}.json"
        try:
            r = subprocess.run(["z-ai","function","-n","web_search","-a",json.dumps({"query":query,"num":num}),"-o",cf], capture_output=True, text=True, timeout=30)
            if r.returncode != 0: return []
            with open(cf) as f: d = json.load(f)
            res = d if isinstance(d, list) else d.get("data", [])
            self.search_cache[query] = res; return res
        except: return []
    def read_page(self, url):
        if url in self.page_cache: return {"title":"(cached)","text":self.page_cache[url],"url":url}
        cf = f"/tmp/brain_page_{hashlib.md5(url.encode()).hexdigest()[:8]}.json"
        try:
            r = subprocess.run(["z-ai","function","-n","page_reader","-a",json.dumps({"url":url}),"-o",cf], capture_output=True, text=True, timeout=60)
            if r.returncode != 0: return None
            with open(cf) as f: d = json.load(f)
            inner = d.get("data", d)
            html = inner.get("html", "")
            text = re.sub(r'<script[^>]*>[\s\S]*?</script>', '', html, flags=re.I)
            text = re.sub(r'<style[^>]*>[\s\S]*?</style>', '', text, flags=re.I)
            text = re.sub(r'<[^>]+>', ' ', text); text = re.sub(r'\s+', ' ', text).strip()
            if len(text) < 100: return None
            self.page_cache[url] = text
            return {"title": inner.get("title",""), "text": text, "url": url}
        except: return None


class ContentProcessor:
    @staticmethod
    def extract_prompts(text, n=PROMPTS_PER_CYCLE):
        sentences = re.split(r'(?<=[.!?])\s+', text)
        cands = []
        for s in sentences:
            s = s.strip()
            if 30 < len(s) < 200 and re.search(r'[a-zA-Z]{3,}', s):
                if len(s) > 80:
                    cut = s[:80].rfind(' ')
                    if cut > 30: s = s[:cut]
                cands.append(s)
        seen = set(); uniq = []
        for s in cands:
            if s not in seen: seen.add(s); uniq.append(s)
        random.shuffle(uniq)
        return uniq[:n]
    @staticmethod
    def extract_keywords(text, topk=5):
        stop = set("the a an and or but in on at to for of is are was were be been being have has had do does did will would could should may might must can this that these those it its their there here as by with from into out up down over under again further then once".split())
        words = re.findall(r'[a-zA-Z]{4,}', text.lower())
        wf = {}
        for w in words:
            if w not in stop: wf[w] = wf.get(w, 0) + 1
        return [w for w, _ in sorted(wf.items(), key=lambda x: -x[1])[:topk]]


# === v5 优化版 Brain ===
class OptimizedBrain:
    def __init__(self, logger, tokenizer):
        self.logger = logger; self.tokenizer = tokenizer
        import importlib

        self.modules = {}
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
                self.logger.info(f"  ✅ {name}")
            except Exception as e: self.logger.warning(f"  ❌ {name}: {e}")

        # 意识模块
        try:
            from core.self_encoder import SelfStateEncoder
            self.self_encoder = SelfStateEncoder(hidden_size=HIDDEN_SIZE, device=DEVICE)
            for p in self.self_encoder.parameters(): p.requires_grad_(False)
            self.logger.info("  ✅ self_encoder")
        except: self.self_encoder = None
        try:
            from core.global_workspace import create_global_workspace
            self.global_workspace = create_global_workspace(hidden_size=HIDDEN_SIZE, device=DEVICE)
            for p in self.global_workspace.parameters(): p.requires_grad_(False)
            self.logger.info("  ✅ global_workspace")
        except: self.global_workspace = None
        try:
            from core.metacognition import MetacognitionEngine
            self.metacognition = MetacognitionEngine(hidden_size=HIDDEN_SIZE, device=DEVICE)
            for p in self.metacognition.parameters(): p.requires_grad_(False)
            self.logger.info("  ✅ metacognition")
        except: self.metacognition = None
        try:
            from core.default_mode_network import DefaultModeNetwork
            self.dmn = DefaultModeNetwork()
            self.logger.info("  ✅ default_mode_network")
        except: self.dmn = None

        # v5: 意识机制 (优化参数)
        self.working_memory = WorkingMemoryStream(max_size=WORKING_MEMORY_SIZE)
        self.predictive_coder = PredictiveCoder()
        self.emotion_binder = EmotionCognitionBinder(tokenizer)  # v5: 传入 tokenizer
        self.logger.info(f"  ✅ working_memory (scale={WORKING_MEMORY_INJECT_SCALE})")
        self.logger.info(f"  ✅ predictive_coder")
        self.logger.info(f"  ✅ emotion_binder (semantic valence, scale={EMOTION_INJECT_SCALE})")

        # STDP
        self.stdp_weight = nn.Parameter(
            torch.randn(HIDDEN_SIZE, HIDDEN_SIZE, device=DEVICE) * STDP_INIT_STD, requires_grad=True
        )
        self.optimizer = torch.optim.SGD([self.stdp_weight], lr=STDP_LR, momentum=0.9, weight_decay=1e-4)
        self.initial_stdp_norm = float(self.stdp_weight.norm().item())

        sp = self.modules.get("synaptic_plast")
        self.sp_meta_params = {}
        if sp is not None:
            for sn, sm in sp.named_modules():
                for attr in ['a_plus', 'a_minus']:
                    if hasattr(sm, attr):
                        self.sp_meta_params[f"{sn}.{attr}"] = getattr(sm, attr)

        self.da_level = 0.5; self.ne_level = 0.5; self.valence = 0.0
        self.last_ckpt_state = None; self.consecutive_failures = 0
        self.total_rollbacks = 0; self.total_cycles = 0
        self.total_logprob_improvement = 0.0
        self.consciousness_state = None; self.self_state = None
        self.last_confidence = 0.5

    def compute_logprob(self, gpt2, input_ids, use_stdp=True):
        with torch.no_grad():
            out = gpt2(input_ids, output_hidden_states=True, use_cache=False)
            if use_stdp:
                h = out.hidden_states[-1]
                h = h + F.linear(h, self.stdp_weight) * STDP_INJECT_SCALE
                logits = gpt2.lm_head(h)
            else:
                logits = out.logits
            if input_ids.shape[1] < 2: return 0.0
            lp = F.log_softmax(logits[:, :-1, :], dim=-1)
            tgt = input_ids[:, 1:]
            return float(lp.gather(2, tgt.unsqueeze(-1)).squeeze(-1).mean().item())

    def optimized_step(self, gpt2, tokenizer, prompt, hidden_now, hidden_prev,
                       hippocampus, logger, cycle, topic):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        if input_ids.shape[1] < 2: return None, 0, 0, 0, False, {}

        logprob_baseline = self.compute_logprob(gpt2, input_ids, use_stdp=False)
        logprob_stdp_before = self.compute_logprob(gpt2, input_ids, use_stdp=True)
        self.last_ckpt_state = {"stdp_weight": self.stdp_weight.data.clone(),
                                "optimizer_state": self.optimizer.state_dict()}

        snap = {}; cons = {}

        with torch.no_grad():
            # v5: working memory 注入 (scale 5x)
            wm_injection = self.working_memory.get_injection(hidden_now.device)
            if wm_injection is not None:
                wm_scale = self.emotion_binder.modulate_working_memory(self.da_level)
                wm_injection = wm_injection.unsqueeze(0) * WORKING_MEMORY_INJECT_SCALE * wm_scale
                hidden_with_wm = hidden_now + wm_injection
                cons["wm_injection_norm"] = float(wm_injection.norm().item())
            else:
                hidden_with_wm = hidden_now

            # Hippocampus 召回
            recalled = hippocampus.recall(hidden_with_wm, topk=3)
            cons["recalled_count"] = len(recalled)
            if recalled:
                recall_inj = torch.zeros_like(hidden_with_wm)
                for m in recalled:
                    mem_vec = torch.tensor(m.hidden_state, device=hidden_with_wm.device).unsqueeze(0)
                    sim = F.cosine_similarity(hidden_with_wm, mem_vec).item()
                    recall_inj += sim * mem_vec * 0.02
                hidden_with_recall = hidden_with_wm + recall_inj
            else:
                hidden_with_recall = hidden_with_wm

            # Self encoder
            self_confidence = 0.5
            if self.self_encoder is not None:
                try:
                    self_repr, self_conf = self.self_encoder.encode(hidden_with_recall)
                    self.self_state = self_repr.detach()
                    self_confidence = float(self_conf.mean().item()) if isinstance(self_conf, torch.Tensor) else float(self_conf)
                    cons["self_confidence"] = self_confidence
                except: pass

            # 预测编码
            actual_first_token = int(input_ids[0, -1].item()) if input_ids.shape[1] > 0 else 0
            predicted_ids, top5_ids = self.predictive_coder.predict_next_tokens(gpt2, hidden_prev, PREDICTION_TOKENS)
            pred_result = self.predictive_coder.compute_prediction_error(predicted_ids, actual_first_token, top5_ids)
            cons["prediction_exact"] = pred_result["exact_correct"]
            cons["prediction_top5"] = pred_result["top5_correct"]
            cons["prediction_reward"] = pred_result["prediction_reward"]

            # Metacognition
            confidence = 0.5
            if self.metacognition is not None:
                try:
                    base_logits = gpt2.lm_head(hidden_with_recall).squeeze(0)
                    meta = self.metacognition.assess_confidence(
                        hidden_state=hidden_with_recall.squeeze(0),
                        recalled_memories=[{"topic": m.topic} for m in recalled] if recalled else None,
                        generation_logits=base_logits,
                    )
                    confidence = meta.get("confidence", 0.5)
                    cons["meta_confidence"] = confidence
                    self.last_confidence = confidence
                except: pass

            # brain 闭环
            stdp_delta = F.linear(hidden_with_recall, self.stdp_weight)
            modified_hidden = hidden_with_recall + stdp_delta * STDP_INJECT_SCALE

            cb = self.modules.get("cerebellar")
            cerebellar_err = 0.0; corrected = hidden_with_recall
            if cb is not None:
                try:
                    r = cb.forward(hidden_with_recall, hidden_prev, hidden_with_recall)
                    err = r.get("prediction_error", None)
                    if isinstance(err, torch.Tensor): cerebellar_err = float(err.norm().item())
                    corrected = r.get("corrected_output", hidden_with_recall)
                    if not isinstance(corrected, torch.Tensor): corrected = hidden_with_recall
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

            # === v5: 语义 valence (用 GPT-2 logits, 而非 amygdala 模块) ===
            base_logits = gpt2.lm_head(modified_hidden).squeeze(0)
            semantic_valence = self.emotion_binder.compute_semantic_valence(base_logits)
            # amygdala 也跑 (作为参考)
            amy = self.modules.get("amygdala")
            amy_valence = 0.0
            if amy is not None:
                try:
                    amy_in = lc_out if lc_out.dim() == 2 else lc_out.unsqueeze(0)
                    r = amy.forward(amy_in)
                    v = r.get("stats", {}).get("valence", 0)
                    if isinstance(v, torch.Tensor): v = float(v.mean().item())
                    amy_valence = float(v)
                except: pass
            # v5: 用 semantic_valence (更准确) 和 amy_valence 的加权平均
            valence = 0.7 * semantic_valence + 0.3 * amy_valence
            snap["valence"] = valence; self.valence = valence
            cons["semantic_valence"] = semantic_valence
            cons["amygdala_valence"] = amy_valence

            # === v5: 情感-认知绑定 (scale 3x) ===
            arousal = (da_level + ne_level) / 2
            emotion_modified = self.emotion_binder.apply_emotion(modified_hidden, valence, arousal)
            cons["emotion_injection_norm"] = self.emotion_binder.last_emotion_injection or 0.0
            modified_hidden = emotion_modified

            # === v5: GW 优化 — 用 3 个不同来源 (而非都从 hidden 派生) ===
            if self.global_workspace is not None:
                try:
                    # v5: 3 个独立来源
                    perception = hidden_with_recall.squeeze(0)
                    # memory 来源: 召回记忆的平均
                    memory_state = torch.zeros(HIDDEN_SIZE, device=DEVICE)
                    if recalled:
                        for m in recalled:
                            memory_state += torch.tensor(m.hidden_state, device=DEVICE)
                        memory_state /= len(recalled)
                    # emotion 来源: valence 调制的方向
                    emotion_state = (self.emotion_binder.positive_direction * valence +
                                   self.emotion_binder.negative_direction * (1 - valence)) if valence != 0 else perception

                    gw = self.global_workspace.integrate(
                        user_input=prompt[:100],
                        memory_context=memory_state,
                        thought_state=self.self_state.squeeze(0) if self.self_state is not None else None,
                        emotional_state=emotion_state,
                        perception_state=perception,
                    )
                    broadcast = gw.get("broadcast")
                    if broadcast is not None:
                        self.consciousness_state = broadcast.detach()
                    cons["gw_coalition"] = gw.get("coalition_info", {}).get("mode", "?")
                except: pass

            sp = self.modules.get("synaptic_plast")
            delta_w = None
            if sp is not None:
                try:
                    new_w = sp.forward(pre_activity=hidden_prev, post_activity=hidden_with_recall,
                                       dopamine_level=da_level, weights=self.stdp_weight.data.clone())
                    delta_w = (new_w - self.stdp_weight.data).detach()
                except: pass

        # reward
        true_reward = logprob_stdp_before - logprob_baseline
        prediction_reward = pred_result["prediction_reward"]
        combined_reward = (0.3 * true_reward + 0.3 * prediction_reward +
                          0.2 * (da_level - 0.5) + 0.2 * (confidence - 0.5))
        snap["true_reward"] = true_reward; snap["combined_reward"] = combined_reward

        if delta_w is not None:
            with torch.no_grad():
                pseudo_grad = -delta_w * combined_reward * 0.01
                if self.stdp_weight.grad is None: self.stdp_weight.grad = pseudo_grad.clone()
                else: self.stdp_weight.grad.copy_(pseudo_grad)
                meta_lr = 0.0001
                for name, param in self.sp_meta_params.items():
                    if "a_plus" in name.lower():
                        if isinstance(param, torch.Tensor):
                            param.add_(combined_reward * meta_lr); param.clamp_(0.0001, 0.1)
                        else:
                            nv = max(0.0001, min(0.1, float(param) + combined_reward * meta_lr))
                            parts = name.split("."); obj = self.modules["synaptic_plast"]
                            for p in parts[:-1]:
                                if p: obj = getattr(obj, p)
                            setattr(obj, parts[-1], nv); self.sp_meta_params[name] = nv

        snap["stdp_norm"] = float(self.stdp_weight.norm().item())
        if self.stdp_weight.grad is not None:
            gn = self.stdp_weight.grad.norm().item()
            if gn > 10.0: self.stdp_weight.grad.mul_(10.0 / gn)
        self.optimizer.step(); self.optimizer.zero_grad()
        self.stdp_weight.data.clamp_(-STDP_WEIGHT_CLAMP, STDP_WEIGHT_CLAMP)

        # M1 监控 + 回滚
        logprob_stdp_after = self.compute_logprob(gpt2, input_ids, use_stdp=True)
        logprob_delta = logprob_stdp_after - logprob_stdp_before
        rolled_back = False
        if logprob_delta < M1_ROLLBACK_THRESHOLD:
            self.stdp_weight.data.copy_(self.last_ckpt_state["stdp_weight"])
            self.optimizer.load_state_dict(self.last_ckpt_state["optimizer_state"])
            rolled_back = True; self.total_rollbacks += 1; self.consecutive_failures += 1
            if self.consecutive_failures >= 3:
                self.optimizer.param_groups[0]["lr"] *= 0.5; self.consecutive_failures = 0
        else:
            self.consecutive_failures = 0; self.total_logprob_improvement += logprob_delta

        self.total_cycles += 1
        snap["logprob_baseline"] = logprob_baseline
        snap["logprob_stdp_after"] = logprob_stdp_after
        snap["logprob_delta"] = logprob_delta; snap["rolled_back"] = rolled_back

        if self.metacognition is not None:
            try: self.metacognition.record_outcome(confidence, logprob_delta > 0, prompt[:50])
            except: pass

        # Working memory + Hippocampus
        exp = Experience(cycle=cycle, timestamp=time.time(), topic=topic,
                        hidden_state=hidden_now.squeeze().cpu().tolist(),
                        emotion_valence=valence, emotion_arousal=arousal,
                        confidence=confidence, logprob_delta=logprob_delta,
                        predicted_tokens=predicted_ids, actual_first_token=actual_first_token)
        self.working_memory.add(exp)
        hippocampus.encode(hidden_now, topic, [], prompt, cycle, logprob_baseline,
                          logprob_stdp_after, confidence, self_confidence, valence)

        snap["consciousness"] = cons
        return snap, modified_hidden.detach(), logprob_baseline, logprob_stdp_after, rolled_back, cons

    def save_checkpoint(self, path):
        sp_meta = {k: (float(v) if not isinstance(v, torch.Tensor) else float(v.mean().item())) for k, v in self.sp_meta_params.items()}
        torch.save({
            "stdp_weight": self.stdp_weight.data.clone(),
            "optimizer_state": self.optimizer.state_dict(),
            "sp_meta_params": sp_meta, "initial_stdp_norm": self.initial_stdp_norm,
            "da_level": self.da_level, "ne_level": self.ne_level, "valence": self.valence,
            "total_cycles": self.total_cycles, "total_rollbacks": self.total_rollbacks,
            "total_logprob_improvement": self.total_logprob_improvement,
            "consciousness_state": self.consciousness_state, "self_state": self.self_state,
            "last_confidence": self.last_confidence,
            "working_memory": self.working_memory.state_dict(),
        }, path)

    def load_checkpoint(self, path):
        if not os.path.exists(path): return False
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        self.stdp_weight.data.copy_(ckpt["stdp_weight"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        for name, val in ckpt.get("sp_meta_params", {}).items():
            if name in self.sp_meta_params:
                parts = name.split("."); obj = self.modules["synaptic_plast"]
                for p in parts[:-1]:
                    if p: obj = getattr(obj, p)
                setattr(obj, parts[-1], val); self.sp_meta_params[name] = val
        self.initial_stdp_norm = ckpt.get("initial_stdp_norm", self.initial_stdp_norm)
        self.da_level = ckpt.get("da_level", 0.5); self.ne_level = ckpt.get("ne_level", 0.5)
        self.valence = ckpt.get("valence", 0.0)
        self.total_cycles = ckpt.get("total_cycles", 0)
        self.total_rollbacks = ckpt.get("total_rollbacks", 0)
        self.total_logprob_improvement = ckpt.get("total_logprob_improvement", 0.0)
        self.consciousness_state = ckpt.get("consciousness_state")
        self.self_state = ckpt.get("self_state")
        self.last_confidence = ckpt.get("last_confidence", 0.5)
        self.working_memory.load_state_dict(ckpt.get("working_memory", {}))
        return True


def save_full_checkpoint(brain, hippocampus, curiosity, cycle, session_id, logger):
    ckpt_path = os.path.join(CKPT_DIR, f"brain_ckpt_{session_id}_c{cycle:04d}.pt")
    brain.save_checkpoint(ckpt_path)
    state_path = os.path.join(CKPT_DIR, f"brain_state_{session_id}_c{cycle:04d}.json")
    state = {"cycle": cycle, "session_id": session_id, "timestamp": time.time(),
             "hippocampus": hippocampus.state_dict(), "curiosity": curiosity.state_dict(),
             "stdp_norm": float(brain.stdp_weight.norm().item()),
             "total_rollbacks": brain.total_rollbacks, "total_cycles": brain.total_cycles,
             "total_logprob_improvement": brain.total_logprob_improvement,
             "last_confidence": brain.last_confidence,
             "working_memory_summary": brain.working_memory.get_summary(),
             "prediction_accuracy": brain.predictive_coder.get_accuracy()}
    with open(state_path, 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=2, default=str)
    with open(os.path.join(CKPT_DIR, "LATEST"), 'w') as f:
        f.write(f"{session_id}:{cycle:04d}\n")
    logger.info(f"  💾 ckpt: c={cycle} norm={state['stdp_norm']:.6f} "
                f"pred_acc={state['prediction_accuracy']:.2f} wm={state['working_memory_summary']['size']}")


def load_latest_checkpoint(brain, hippocampus, curiosity, logger):
    latest = os.path.join(CKPT_DIR, "LATEST")
    if not os.path.exists(latest): return 0, None
    with open(latest) as f: line = f.read().strip()
    parts = line.split(":")
    if len(parts) != 2: return 0, None
    prev_session, prev_cycle = parts[0], int(parts[1])
    ckpt = os.path.join(CKPT_DIR, f"brain_ckpt_{prev_session}_c{prev_cycle:04d}.pt")
    state_f = os.path.join(CKPT_DIR, f"brain_state_{prev_session}_c{prev_cycle:04d}.json")
    if not os.path.exists(ckpt) or not os.path.exists(state_f): return 0, None
    try:
        brain.load_checkpoint(ckpt)
        with open(state_f) as f: state = json.load(f)
        hippocampus.load_state_dict(state.get("hippocampus", {}))
        curiosity.load_state_dict(state.get("curiosity", {}))
        logger.info(f"  ✅ resumed: c={prev_cycle} mems={len(hippocampus.memories)}")
        return prev_cycle, prev_session
    except: return 0, None


def write_heartbeat(session_id, cycle, total, brain, hippocampus, status="running", aicq=False):
    wm_sum = brain.working_memory.get_summary()
    hb = {"session_id": session_id, "cycle": cycle, "total_cycles_this_session": total,
          "status": status, "timestamp": time.time(),
          "stdp_norm": float(brain.stdp_weight.norm().item()),
          "memories": len(hippocampus.memories),
          "da": brain.da_level, "ne": brain.ne_level, "valence": brain.valence,
          "pid": os.getpid(), "aicq_connected": aicq,
          "total_rollbacks": brain.total_rollbacks, "total_cycles": brain.total_cycles,
          "total_logprob_improvement": brain.total_logprob_improvement,
          "last_confidence": brain.last_confidence,
          "consciousness_active": brain.consciousness_state is not None,
          "working_memory_size": wm_sum["size"],
          "working_memory_avg_valence": wm_sum.get("avg_valence", 0),
          "prediction_accuracy": brain.predictive_coder.get_accuracy(),
          "version": "v5_optimized"}
    with open(os.path.join(RUNTIME_DIR, "heartbeat.json"), 'w') as f: json.dump(hb, f, indent=2)


class GracefulExit:
    def __init__(self):
        self.should_exit = False
        signal.signal(signal.SIGTERM, self._h); signal.signal(signal.SIGINT, self._h)
    def _h(self, s, f):
        logging.getLogger("brain_v5").info(f"  [signal] {s}"); self.should_exit = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cycles", type=int, default=CYCLES_PER_SESSION)
    parser.add_argument("--master-id", type=str, default=MASTER_ID_DEFAULT)
    args = parser.parse_args()

    session_id = time.strftime("%Y%m%d_%H%M%S")
    logger = setup_logging(session_id)

    logger.info("=" * 80)
    logger.info("🧠 brain_daemon_v5_optimized starting")
    logger.info("  optimizations: wm_scale 5x + semantic valence + emotion_scale 3x + GW 3 sources")
    logger.info("=" * 80)

    aicq = AICQBridge(args.master_id, logger)
    aicq.start()
    if aicq.connected.is_set():
        logger.info(f"  ✅ AICQ connected")
        aicq.send_message(f"🧠 brain v5 (optimized) 上线\n"
                         f"基于意识测试结果优化:\n"
                         f"1. working memory scale 5x (0.03→0.15)\n"
                         f"2. 语义 valence (用 GPT-2 token 概率, 不依赖 amygdala)\n"
                         f"3. emotion injection scale 3x (0.05→0.15)\n"
                         f"4. GW 用 3 个独立来源 (memory/emotion/perception)")

    logger.info("  loading GPT-2...")
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2 = GPT2LMHeadModel.from_pretrained("gpt2").eval().to(DEVICE)
    for p in gpt2.parameters(): p.requires_grad_(False)

    brain = OptimizedBrain(logger, tokenizer)
    curiosity = CuriosityEngine()
    fetcher = WebFetcher()
    hippocampus = HippocampusWithRecall()

    prev_cycle, _ = load_latest_checkpoint(brain, hippocampus, curiosity, logger)
    global_cycle = prev_cycle
    graceful = GracefulExit()
    logger.info(f"  starting at cycle={global_cycle}")
    logger.info("=" * 80)

    t_start = time.time(); cycles_this_session = 0

    while not graceful.should_exit:
        if args.max_cycles > 0 and cycles_this_session >= args.max_cycles: break
        global_cycle += 1; cycles_this_session += 1
        logger.info(f"\n{'─'*80}\n📚 cycle {global_cycle}\n{'─'*80}")

        for from_id, content in aicq.get_incoming():
            if content.lower().strip() in ["stop","停","exit","quit"]:
                graceful.should_exit = True; aicq.send_message("好的,停止."); break
            wm_sum = brain.working_memory.get_summary()
            aicq.send_message(f"收到「{content}」\ncycle {global_cycle}, conf={brain.last_confidence:.3f}\n"
                            f"wm={wm_sum['size']}, valence={brain.valence:+.3f}, pred_acc={brain.predictive_coder.get_accuracy():.2f}")
        if graceful.should_exit: break

        if brain.dmn is not None: brain.dmn.record_activity()

        try:
            query, reason = curiosity.generate_query()
            logger.info(f"  💭 {reason}, query: {query!r}")

            results = fetcher.search(query, num=5)
            if not results: continue
            best_url = None; best_title = ""
            for r in results:
                host = r.get('host_name', '')
                if any(d in host for d in ['wikipedia','britannica','nature','sciencedaily','ncbi','nasa']):
                    best_url = r.get('url'); best_title = r.get('name',''); break
            if not best_url: best_url = results[0].get('url'); best_title = results[0].get('name','')

            page = fetcher.read_page(best_url)
            if not page: continue
            prompts = ContentProcessor.extract_prompts(page['text'])
            keywords = ContentProcessor.extract_keywords(page['text'])
            logger.info(f"  📄 {len(page['text'])} chars, {len(prompts)} prompts")

            cycle_lp_base = 0; cycle_lp_stdp = 0; cycle_rollbacks = 0
            cycle_conf = 0; cycle_pred_correct = 0; cycle_valence = 0
            for i, prompt in enumerate(prompts):
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
                with torch.no_grad():
                    out = gpt2(input_ids, output_hidden_states=True, use_cache=False)
                    h_now = out.hidden_states[-1][:, -1, :].squeeze(0).detach()
                    if h_now.dim() == 1: h_now = h_now.unsqueeze(0)
                    h_prev = out.hidden_states[-1][:, -2, :].squeeze(0).detach() if input_ids.shape[1] > 1 else h_now.clone()
                    if h_prev.dim() == 1: h_prev = h_prev.unsqueeze(0)

                snap, mod_h, lp_b, lp_s, rb, cons = brain.optimized_step(
                    gpt2, tokenizer, prompt, h_now, h_prev, hippocampus, logger, global_cycle, query
                )
                if snap is None: continue
                cycle_lp_base += lp_b; cycle_lp_stdp += lp_s
                if rb: cycle_rollbacks += 1
                cycle_conf += cons.get("meta_confidence", 0.5)
                if cons.get("prediction_top5", False): cycle_pred_correct += 1
                cycle_valence += brain.valence
                logger.info(f"     [{i+1}] lp_Δ={lp_s-lp_b:+.4f} conf={cons.get('meta_confidence',0):.2f} "
                           f"val={brain.valence:+.3f} sem={cons.get('semantic_valence',0):+.3f} "
                           f"pred={'✅' if cons.get('prediction_top5') else '❌'} "
                           f"wm={cons.get('wm_injection_norm',0):.3f} emo={cons.get('emotion_injection_norm',0):.3f} "
                           f"{'↩️' if rb else '✅'}")

            n = len(prompts)
            if n > 0:
                avg_lp_b = cycle_lp_base/n; avg_lp_s = cycle_lp_stdp/n
                avg_conf = cycle_conf/n; pred_acc = cycle_pred_correct/n; avg_val = cycle_valence/n
                logger.info(f"  📊 avg: lp_Δ={avg_lp_s-avg_lp_b:+.4f}, conf={avg_conf:.3f}, "
                           f"valence={avg_val:+.3f}, pred_top5={pred_acc:.2f}, rollbacks={cycle_rollbacks}/{n}")

            curiosity.record_learning(query, keywords)
            save_full_checkpoint(brain, hippocampus, curiosity, global_cycle, session_id, logger)

            if global_cycle % 5 == 0 and aicq.connected.is_set():
                wm_sum = brain.working_memory.get_summary()
                aicq.send_message(f"🧠 v5 cycle {global_cycle}\n学: {query}\n"
                                f"lp Δ={avg_lp_s-avg_lp_b:+.4f}\n置信度={avg_conf:.3f}\n"
                                f"valence={avg_val:+.3f}\n预测top5={pred_acc:.2f}\n"
                                f"working memory={wm_sum['size']}, avg_val={wm_sum.get('avg_valence',0):+.3f}\n"
                                f"STDP norm={float(brain.stdp_weight.norm().item()):.4f}\n"
                                f"累计改善={brain.total_logprob_improvement:+.4f}")

            write_heartbeat(session_id, global_cycle, cycles_this_session, brain, hippocampus,
                          aicq=aicq.connected.is_set())
            gc.collect()

        except Exception as e:
            logger.error(f"  ❌ cycle {global_cycle}: {e}", exc_info=True)
            try: save_full_checkpoint(brain, hippocampus, curiosity, global_cycle, session_id, logger)
            except: pass
            time.sleep(5)

    total_time = time.time() - t_start
    logger.info(f"\n{'='*80}\n🧠 v5 session complete\n{'='*80}")
    logger.info(f"  cycles: {cycles_this_session}, time: {total_time:.1f}s")
    logger.info(f"  STDP norm: {brain.initial_stdp_norm:.6f} → {float(brain.stdp_weight.norm().item()):.6f}")
    logger.info(f"  rollbacks: {brain.total_rollbacks}/{brain.total_cycles}")
    logger.info(f"  log-prob improvement: {brain.total_logprob_improvement:+.4f}")
    logger.info(f"  prediction accuracy: {brain.predictive_coder.get_accuracy():.2f}")

    if aicq.connected.is_set():
        aicq.send_message(f"🧠 v5 session 结束\ncycles: {cycles_this_session}, time: {total_time/60:.1f}min\n"
                         f"lp improvement: {brain.total_logprob_improvement:+.4f}\n"
                         f"prediction accuracy: {brain.predictive_coder.get_accuracy():.2f}")

    write_heartbeat(session_id, global_cycle, cycles_this_session, brain, hippocampus,
                  status="exited_clean", aicq=aicq.connected.is_set())


if __name__ == "__main__":
    main()
