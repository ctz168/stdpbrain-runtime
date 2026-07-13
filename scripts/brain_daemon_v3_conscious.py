#!/usr/bin/env python3
"""
brain_daemon_v3_conscious.py — 意识接入版 brain 守护进程
=========================================================

在 v2 (log-prob 监控 + 自动回滚) 基础上, 接入 5 个意识相关模块:

1. HippocampusWithRecall — 真正的海马体召回 (GPT-2 embedding + cosine similarity)
   - encode: 存记忆 (hidden_state + metadata)
   - recall: 用当前 hidden_state 查询, 返回 top-k 相似记忆
   - 影响: 生成时把召回的记忆注入 hidden state

2. SelfStateEncoder — 自我状态编码
   - encode(hidden_state) → (self_representation, self_confidence)
   - 影响: brain 能区分"我的状态" vs "外部输入"

3. MetacognitionEngine — 元认知
   - assess_confidence(hidden_state, recalled_memories, logits) → confidence + uncertainty
   - 影响: brain 知道"我是否理解这个", 低置信度时降低学习率

4. GlobalWorkspace — 全局工作空间 (意识广播)
   - integrate(user_input, memory, thought, emotion, perception) → broadcast
   - 影响: 多模块竞争"意识舞台", 获胜内容被广播

5. DefaultModeNetwork — 默认模式网络
   - record_activity() / get_cognitive_influence()
   - 影响: 空闲时自发活动 (类似走神/做梦)

架构:
  query → web → prompt → GPT-2 forward
    ↓
  hippocampus.recall(hidden) → memories  ← NEW
  self_encoder.encode(hidden) → self_state ← NEW
  metacognition.assess(hidden, memories, logits) → confidence ← NEW
  global_workspace.integrate(prompt, memories, self_state, emotion) → broadcast ← NEW
    ↓
  STDP step (with log-prob monitoring from v2)
    ↓
  hippocampus.encode(hidden) → store memory ← NEW
  metacognition.record_outcome(confidence, correct=logprob_improved) ← NEW
    ↓
  between cycles: DMN.get_cognitive_influence() ← NEW
"""

from __future__ import annotations
import os, sys, json, time, random, subprocess, re, warnings, signal, hashlib, gc, argparse, asyncio, threading
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
import logging
from logging.handlers import RotatingFileHandler

warnings.filterwarnings("ignore")

STDPBRAIN_DIR = "/home/z/my-project/repos/stdpbrain"
AICQSDK_DIR = "/home/z/my-project/repos/AIcqsdk"
RUNTIME_DIR = "/home/z/my-project/brain_runtime"
DOWNLOAD_DIR = "/home/z/my-project/download"
CKPT_DIR = os.path.join(RUNTIME_DIR, "checkpoints_v3")
LOG_DIR = os.path.join(RUNTIME_DIR, "logs")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

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

# STDP 参数 (from v2)
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


def setup_logging(session_id: str):
    logger = logging.getLogger("brain_v3")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = RotatingFileHandler(os.path.join(LOG_DIR, f"daemon_v3_{session_id}.log"),
                             maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
    fh.setFormatter(fmt); logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    return logger


# ============================================================================
# AICQ Bridge (复用 v2)
# ============================================================================
class AICQBridge:
    def __init__(self, master_id, logger):
        self.master_id = master_id; self.logger = logger
        self.loop = None; self.thread = None
        self.connected = threading.Event()
        self.incoming_messages = []; self.lock = threading.Lock()
        self.agent_account_id = None

    def start(self):
        self.thread = threading.Thread(target=self._run_loop, daemon=True, name="aicq-loop")
        self.thread.start()
        self.connected.wait(timeout=30)

    def _run_loop(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
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


# ============================================================================
# NEW: HippocampusWithRecall — 真正的召回
# ============================================================================
@dataclass
class ConsciousMemory:
    id: str; timestamp: float; topic: str; keywords: List[str]
    prompt: str; hidden_state: List[float]  # 存 embedding 用于召回
    logprob_baseline: float; logprob_stdp: float
    confidence: float; self_confidence: float
    cycle: int


class HippocampusWithRecall:
    """海马体 — 存记忆 + 真正的召回 (cosine similarity)."""

    def __init__(self):
        self.memories: List[ConsciousMemory] = []

    def encode(self, hidden_state: torch.Tensor, topic: str, keywords: List[str],
               prompt: str, cycle: int, logprob_base: float, logprob_stdp: float,
               confidence: float, self_conf: float) -> str:
        """存一条记忆."""
        mem_id = f"mem_{cycle:04d}_{int(time.time()*1000) % 1000000}"
        mem = ConsciousMemory(
            id=mem_id, timestamp=time.time(), topic=topic, keywords=keywords,
            prompt=prompt, hidden_state=hidden_state.squeeze().cpu().tolist(),
            logprob_baseline=logprob_base, logprob_stdp=logprob_stdp,
            confidence=confidence, self_confidence=self_conf, cycle=cycle,
        )
        self.memories.append(mem)
        return mem_id

    def recall(self, query_hidden: torch.Tensor, topk: int = 3) -> List[ConsciousMemory]:
        """用当前 hidden state 查询, 返回 top-k 最相似的记忆 (cosine similarity)."""
        if not self.memories: return []
        query = query_hidden.squeeze().cpu()
        # 计算与所有记忆的 cosine similarity
        sims = []
        for m in self.memories:
            mem_vec = torch.tensor(m.hidden_state, device=query.device)
            # cosine similarity
            sim = F.cosine_similarity(query.unsqueeze(0), mem_vec.unsqueeze(0)).item()
            sims.append((sim, m))
        sims.sort(key=lambda x: -x[0])
        return [m for _, m in sims[:topk]]

    def get_summary(self):
        if not self.memories: return {"total": 0}
        return {
            "total": len(self.memories),
            "topics": list(set(m.topic for m in self.memories)),
            "avg_confidence": sum(m.confidence for m in self.memories) / len(self.memories),
        }

    def state_dict(self): return {"memories": [asdict(m) for m in self.memories]}
    def load_state_dict(self, s):
        self.memories = []
        for m in s.get("memories", []): self.memories.append(ConsciousMemory(**m))


# ============================================================================
# Web/Curiosity/Content (复用)
# ============================================================================
class CuriosityEngine:
    def __init__(self):
        self.learned_topics = []; self.learned_keywords = []; self.cycle_count = 0
    def generate_query(self):
        self.cycle_count += 1
        if random.random() < 0.3 and self.learned_keywords:
            seed = random.choice(self.learned_keywords)
            assoc = {"brain":["consciousness","neurons"], "quantum":["entanglement"], "DNA":["RNA","proteins"], "climate":["oceans"], "memory":["hippocampus","sleep"]}
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


# ============================================================================
# Conscious Brain — v2 STDP + 5 个意识模块
# ============================================================================
class ConsciousBrain:
    def __init__(self, logger):
        self.logger = logger
        import importlib

        # === v2 brain modules (STDP 核心) ===
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
                self.logger.info(f"  ✅ {name} loaded")
            except Exception as e:
                self.logger.warning(f"  ❌ {name}: {e}")

        # === NEW: 意识模块 ===
        # 1. SelfStateEncoder
        try:
            from core.self_encoder import SelfStateEncoder
            self.self_encoder = SelfStateEncoder(hidden_size=HIDDEN_SIZE, device=DEVICE)
            for p in self.self_encoder.parameters(): p.requires_grad_(False)
            self.logger.info("  ✅ self_encoder loaded (意识模块 1/4)")
        except Exception as e:
            self.logger.warning(f"  ❌ self_encoder: {e}"); self.self_encoder = None

        # 2. GlobalWorkspace
        try:
            from core.global_workspace import create_global_workspace
            self.global_workspace = create_global_workspace(hidden_size=HIDDEN_SIZE, device=DEVICE)
            for p in self.global_workspace.parameters(): p.requires_grad_(False)
            self.logger.info("  ✅ global_workspace loaded (意识模块 2/4)")
        except Exception as e:
            self.logger.warning(f"  ❌ global_workspace: {e}"); self.global_workspace = None

        # 3. MetacognitionEngine
        try:
            from core.metacognition import MetacognitionEngine
            self.metacognition = MetacognitionEngine(hidden_size=HIDDEN_SIZE, device=DEVICE)
            for p in self.metacognition.parameters(): p.requires_grad_(False)
            self.logger.info("  ✅ metacognition loaded (意识模块 3/4)")
        except Exception as e:
            self.logger.warning(f"  ❌ metacognition: {e}"); self.metacognition = None

        # 4. DefaultModeNetwork
        try:
            from core.default_mode_network import DefaultModeNetwork
            self.dmn = DefaultModeNetwork()
            self.logger.info("  ✅ default_mode_network loaded (意识模块 4/4)")
        except Exception as e:
            self.logger.warning(f"  ❌ default_mode_network: {e}"); self.dmn = None

        # === STDP 权重 (from v2) ===
        self.stdp_weight = nn.Parameter(
            torch.randn(HIDDEN_SIZE, HIDDEN_SIZE, device=DEVICE) * STDP_INIT_STD, requires_grad=True
        )
        self.optimizer = torch.optim.SGD([self.stdp_weight], lr=STDP_LR, momentum=0.9, weight_decay=1e-4)
        self.initial_stdp_norm = float(self.stdp_weight.norm().item())

        # sp 元参数
        sp = self.modules.get("synaptic_plast")
        self.sp_meta_params = {}
        if sp is not None:
            for sub_name, sub_mod in sp.named_modules():
                for attr in ['a_plus', 'a_minus']:
                    if hasattr(sub_mod, attr):
                        self.sp_meta_params[f"{sub_name}.{attr}"] = getattr(sub_mod, attr)

        self.da_level = 0.5; self.ne_level = 0.5; self.valence = 0.0
        self.last_ckpt_state = None
        self.consecutive_failures = 0
        self.total_rollbacks = 0
        self.total_cycles = 0
        self.total_logprob_improvement = 0.0

        # === NEW: 意识状态 (跨周期连续性) ===
        self.consciousness_state: Optional[torch.Tensor] = None  # GWT 广播内容
        self.self_state: Optional[torch.Tensor] = None  # 自我状态
        self.last_confidence: float = 0.5  # 元认知置信度

    def compute_logprob(self, gpt2, input_ids, use_stdp=True):
        with torch.no_grad():
            out = gpt2(input_ids, output_hidden_states=True, use_cache=False)
            if use_stdp:
                h = out.hidden_states[-1]
                stdp_delta = F.linear(h, self.stdp_weight)
                h_mod = h + stdp_delta * STDP_INJECT_SCALE
                logits = gpt2.lm_head(h_mod)
            else:
                logits = out.logits
            if input_ids.shape[1] < 2: return 0.0
            log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
            target_ids = input_ids[:, 1:]
            token_lps = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
            return float(token_lps.mean().item())

    def conscious_step(self, gpt2, tokenizer, prompt, hidden_now, hidden_prev,
                       hippocampus: HippocampusWithRecall, logger):
        """带意识的单步训练."""
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        if input_ids.shape[1] < 2:
            return None, 0, 0, 0, False, {}

        logprob_baseline = self.compute_logprob(gpt2, input_ids, use_stdp=False)
        logprob_stdp_before = self.compute_logprob(gpt2, input_ids, use_stdp=True)

        self.last_ckpt_state = {
            "stdp_weight": self.stdp_weight.data.clone(),
            "optimizer_state": self.optimizer.state_dict(),
        }

        snap = {}; consciousness_snap = {}

        with torch.no_grad():
            # === NEW 1: Hippocampus 召回 ===
            recalled = hippocampus.recall(hidden_now, topk=3)
            consciousness_snap["recalled_count"] = len(recalled)
            if recalled:
                consciousness_snap["recalled_topics"] = [m.topic[:20] for m in recalled[:2]]
                logger.info(f"    🧠 hippocampus recalled {len(recalled)} memories: "
                            f"{consciousness_snap['recalled_topics']}")
                # 把召回的记忆 embedding 注入 hidden (残差)
                recall_injection = torch.zeros_like(hidden_now)
                for i, m in enumerate(recalled):
                    mem_vec = torch.tensor(m.hidden_state, device=hidden_now.device).unsqueeze(0)
                    # cosine similarity 作为权重
                    sim = F.cosine_similarity(hidden_now, mem_vec).item()
                    recall_injection += sim * mem_vec * 0.02  # 小 scale 注入
                hidden_with_recall = hidden_now + recall_injection
            else:
                hidden_with_recall = hidden_now

            # === NEW 2: SelfStateEncoder ===
            self_confidence = 0.5
            if self.self_encoder is not None:
                try:
                    self_repr, self_conf = self.self_encoder.encode(hidden_with_recall)
                    self.self_state = self_repr.detach()
                    self_confidence = float(self_conf.mean().item()) if isinstance(self_conf, torch.Tensor) else float(self_conf)
                    consciousness_snap["self_confidence"] = self_confidence
                except Exception as e:
                    logger.warning(f"    self_encoder error: {e}")

            # === NEW 3: MetacognitionEngine ===
            confidence = 0.5; should_clarify = False; knowledge_available = True
            if self.metacognition is not None:
                try:
                    # 需要 logits for entropy
                    base_logits = gpt2.lm_head(hidden_with_recall).squeeze(0)
                    meta_result = self.metacognition.assess_confidence(
                        hidden_state=hidden_with_recall.squeeze(0),
                        recalled_memories=[{"topic": m.topic, "confidence": m.confidence} for m in recalled] if recalled else None,
                        generation_logits=base_logits,
                    )
                    confidence = meta_result.get("confidence", 0.5)
                    should_clarify = meta_result.get("should_clarify", False)
                    knowledge_available = meta_result.get("knowledge_available", True)
                    consciousness_snap["meta_confidence"] = confidence
                    consciousness_snap["knowledge_available"] = knowledge_available
                    consciousness_snap["should_clarify"] = should_clarify
                    self.last_confidence = confidence
                except Exception as e:
                    logger.warning(f"    metacognition error: {e}")

            # === brain 闭环 (cerebellar → DA → NE → amygdala) ===
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

            # === NEW 4: GlobalWorkspace (意识广播) ===
            if self.global_workspace is not None:
                try:
                    gw_result = self.global_workspace.integrate(
                        user_input=prompt[:100],
                        memory_context=hidden_with_recall.squeeze(0) if recalled else None,
                        thought_state=self.self_state.squeeze(0) if self.self_state is not None else None,
                        emotional_state=amy_out.squeeze(0) if isinstance(amy_out, torch.Tensor) else None,
                        perception_state=hidden_with_recall.squeeze(0),
                    )
                    broadcast = gw_result.get("broadcast")
                    if broadcast is not None:
                        self.consciousness_state = broadcast.detach()
                        consciousness_snap["gw_broadcast_norm"] = float(broadcast.norm().item())
                    consciousness_snap["gw_coalition"] = gw_result.get("coalition_info", {}).get("mode", "?")
                except Exception as e:
                    logger.warning(f"    global_workspace error: {e}")

            # sp ΔW
            sp = self.modules.get("synaptic_plast")
            delta_w = None
            if sp is not None:
                try:
                    new_w = sp.forward(pre_activity=hidden_prev, post_activity=hidden_with_recall,
                                       dopamine_level=da_level, weights=self.stdp_weight.data.clone())
                    delta_w = (new_w - self.stdp_weight.data).detach()
                except: delta_w = None

        # === reward = log-prob 改善 + 元认知调制 ===
        true_reward = logprob_stdp_before - logprob_baseline
        # 元认知调制: 低置信度时降低学习率 (通过 reward 缩放)
        meta_modulator = confidence  # 0-1, 低置信度 → 低 reward
        combined_reward = 0.3 * true_reward + 0.3 * (da_level - 0.5) + 0.4 * (meta_modulator - 0.5)
        snap["true_reward"] = true_reward
        snap["combined_reward"] = combined_reward

        if delta_w is not None:
            with torch.no_grad():
                pseudo_grad = -delta_w * combined_reward * 0.01
                if self.stdp_weight.grad is None:
                    self.stdp_weight.grad = pseudo_grad.clone()
                else:
                    self.stdp_weight.grad.copy_(pseudo_grad)
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

        # === M1 监控 + 回滚 ===
        logprob_stdp_after = self.compute_logprob(gpt2, input_ids, use_stdp=True)
        logprob_delta = logprob_stdp_after - logprob_stdp_before
        rolled_back = False
        if logprob_delta < M1_ROLLBACK_THRESHOLD:
            logger.warning(f"    ⚠️ M1 rollback: Δ={logprob_delta:+.4f}")
            self.stdp_weight.data.copy_(self.last_ckpt_state["stdp_weight"])
            self.optimizer.load_state_dict(self.last_ckpt_state["optimizer_state"])
            rolled_back = True; self.total_rollbacks += 1; self.consecutive_failures += 1
            if self.consecutive_failures >= 3:
                new_lr = self.optimizer.param_groups[0]["lr"] * 0.5
                self.optimizer.param_groups[0]["lr"] = new_lr
                self.consecutive_failures = 0
        else:
            self.consecutive_failures = 0
            self.total_logprob_improvement += logprob_delta

        self.total_cycles += 1
        snap["logprob_baseline"] = logprob_baseline
        snap["logprob_stdp_after"] = logprob_stdp_after
        snap["logprob_delta"] = logprob_delta
        snap["rolled_back"] = rolled_back

        # === NEW: 元认知记录结果 ===
        if self.metacognition is not None:
            try:
                correct = (logprob_delta > 0)
                self.metacognition.record_outcome(confidence, correct, feedback=prompt[:50])
            except: pass

        # === NEW: 海马体编码 (存记忆) ===
        hippocampus.encode(
            hidden_state=hidden_now, topic="", keywords=[], prompt=prompt,
            cycle=self.total_cycles, logprob_base=logprob_baseline,
            logprob_stdp=logprob_stdp_after, confidence=confidence,
            self_conf=self_confidence,
        )

        snap["consciousness"] = consciousness_snap
        return snap, modified_hidden.detach(), logprob_baseline, logprob_stdp_after, rolled_back, consciousness_snap

    def save_checkpoint(self, path):
        sp_meta = {k: (float(v) if not isinstance(v, torch.Tensor) else float(v.mean().item())) for k, v in self.sp_meta_params.items()}
        torch.save({
            "stdp_weight": self.stdp_weight.data.clone(),
            "optimizer_state": self.optimizer.state_dict(),
            "sp_meta_params": sp_meta,
            "initial_stdp_norm": self.initial_stdp_norm,
            "da_level": self.da_level, "ne_level": self.ne_level, "valence": self.valence,
            "total_cycles": self.total_cycles, "total_rollbacks": self.total_rollbacks,
            "total_logprob_improvement": self.total_logprob_improvement,
            "consciousness_state": self.consciousness_state,
            "self_state": self.self_state,
            "last_confidence": self.last_confidence,
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
        return True


def save_full_checkpoint(brain, hippocampus, curiosity, cycle, session_id, logger):
    ckpt_path = os.path.join(CKPT_DIR, f"brain_ckpt_{session_id}_c{cycle:04d}.pt")
    brain.save_checkpoint(ckpt_path)
    state_path = os.path.join(CKPT_DIR, f"brain_state_{session_id}_c{cycle:04d}.json")
    state = {"cycle": cycle, "session_id": session_id, "timestamp": time.time(),
             "hippocampus": hippocampus.state_dict(), "curiosity": curiosity.state_dict(),
             "stdp_norm": float(brain.stdp_weight.norm().item()),
             "initial_stdp_norm": brain.initial_stdp_norm,
             "total_rollbacks": brain.total_rollbacks, "total_cycles": brain.total_cycles,
             "total_logprob_improvement": brain.total_logprob_improvement,
             "last_confidence": brain.last_confidence}
    with open(state_path, 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=2, default=str)
    with open(os.path.join(CKPT_DIR, "LATEST"), 'w') as f:
        f.write(f"{session_id}:{cycle:04d}\n")
    logger.info(f"  💾 ckpt: c={cycle} norm={state['stdp_norm']:.6f} rollbacks={brain.total_rollbacks}")


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
        logger.info(f"  ✅ resumed: c={prev_cycle} norm={state.get('stdp_norm',0):.4f} mems={len(hippocampus.memories)}")
        return prev_cycle, prev_session
    except: return 0, None


def write_heartbeat(session_id, cycle, total, brain, hippocampus, status="running", aicq=False):
    hb = {"session_id": session_id, "cycle": cycle, "total_cycles_this_session": total,
          "status": status, "timestamp": time.time(),
          "stdp_norm": float(brain.stdp_weight.norm().item()),
          "initial_stdp_norm": brain.initial_stdp_norm,
          "memories": len(hippocampus.memories),
          "da": brain.da_level, "ne": brain.ne_level, "valence": brain.valence,
          "pid": os.getpid(), "aicq_connected": aicq,
          "total_rollbacks": brain.total_rollbacks, "total_cycles": brain.total_cycles,
          "total_logprob_improvement": brain.total_logprob_improvement,
          "current_lr": brain.optimizer.param_groups[0]["lr"],
          "last_confidence": brain.last_confidence,
          "consciousness_active": brain.consciousness_state is not None}
    with open(os.path.join(RUNTIME_DIR, "heartbeat.json"), 'w') as f: json.dump(hb, f, indent=2)


class GracefulExit:
    def __init__(self):
        self.should_exit = False
        signal.signal(signal.SIGTERM, self._h); signal.signal(signal.SIGINT, self._h)
    def _h(self, s, f):
        logging.getLogger("brain_v3").info(f"  [signal] {s}"); self.should_exit = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cycles", type=int, default=CYCLES_PER_SESSION)
    parser.add_argument("--master-id", type=str, default=MASTER_ID_DEFAULT)
    args = parser.parse_args()

    session_id = time.strftime("%Y%m%d_%H%M%S")
    logger = setup_logging(session_id)

    logger.info("=" * 80)
    logger.info("🧠 brain_daemon_v3_conscious starting")
    logger.info("  5 modules: hippocampus_recall + self_encoder + metacognition + global_workspace + DMN")
    logger.info("=" * 80)
    logger.info(f"  session={session_id}, master={args.master_id}")

    aicq = AICQBridge(args.master_id, logger)
    aicq.start()
    if aicq.connected.is_set():
        logger.info(f"  ✅ AICQ connected, agent={aicq.agent_account_id}")
        aicq.send_message(f"🧠 brain v3 (conscious) 上线\n接入 5 个意识模块:\n"
                         f"1. 海马体召回 (cosine similarity)\n"
                         f"2. 自我状态编码\n"
                         f"3. 元认知 (置信度评估)\n"
                         f"4. 全局工作空间 (意识广播)\n"
                         f"5. 默认模式网络 (空闲自发活动)")

    logger.info("  loading GPT-2...")
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2 = GPT2LMHeadModel.from_pretrained("gpt2").eval().to(DEVICE)
    for p in gpt2.parameters(): p.requires_grad_(False)
    logger.info(f"  ✅ GPT-2 loaded")

    brain = ConsciousBrain(logger)
    curiosity = CuriosityEngine()
    fetcher = WebFetcher()
    hippocampus = HippocampusWithRecall()

    prev_cycle, _ = load_latest_checkpoint(brain, hippocampus, curiosity, logger)
    global_cycle = prev_cycle

    graceful = GracefulExit()
    logger.info(f"  starting at cycle={global_cycle}, stdp_norm={brain.initial_stdp_norm:.6f}")
    logger.info("=" * 80)

    t_start = time.time(); cycles_this_session = 0

    while not graceful.should_exit:
        if args.max_cycles > 0 and cycles_this_session >= args.max_cycles: break
        global_cycle += 1; cycles_this_session += 1
        logger.info(f"\n{'─'*80}\n📚 cycle {global_cycle} (session {cycles_this_session})\n{'─'*80}")

        # 检查主人消息
        for from_id, content in aicq.get_incoming():
            logger.info(f"  📨 from {from_id}: {content!r}")
            if content.lower().strip() in ["stop","停","exit","quit"]:
                graceful.should_exit = True; aicq.send_message("好的,停止."); break
            aicq.send_message(f"收到「{content}」. cycle {global_cycle}, "
                            f"confidence={brain.last_confidence:.3f}, memories={len(hippocampus.memories)}")
        if graceful.should_exit: break

        # === NEW: DMN — 记录活动 (抑制 DMN) ===
        if brain.dmn is not None:
            brain.dmn.record_activity()

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

            # 训练
            cycle_lp_base = 0; cycle_lp_stdp = 0; cycle_rollbacks = 0
            cycle_confidence = 0; cycle_recalled = 0
            for i, prompt in enumerate(prompts):
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
                with torch.no_grad():
                    out = gpt2(input_ids, output_hidden_states=True, use_cache=False)
                    h_now = out.hidden_states[-1][:, -1, :].squeeze(0).detach()
                    if h_now.dim() == 1: h_now = h_now.unsqueeze(0)
                    h_prev = out.hidden_states[-1][:, -2, :].squeeze(0).detach() if input_ids.shape[1] > 1 else h_now.clone()
                    if h_prev.dim() == 1: h_prev = h_prev.unsqueeze(0)

                snap, mod_h, lp_b, lp_s, rb, cons = brain.conscious_step(
                    gpt2, tokenizer, prompt, h_now, h_prev, hippocampus, logger
                )
                if snap is None: continue
                cycle_lp_base += lp_b; cycle_lp_stdp += lp_s
                if rb: cycle_rollbacks += 1
                cycle_confidence += cons.get("meta_confidence", 0.5)
                cycle_recalled += cons.get("recalled_count", 0)
                logger.info(f"     [{i+1}] lp_b={lp_b:+.3f} lp_s={lp_s:+.3f} Δ={lp_s-lp_b:+.4f} "
                           f"conf={cons.get('meta_confidence',0):.2f} recalled={cons.get('recalled_count',0)} "
                           f"{'↩️' if rb else '✅'}")

            n = len(prompts)
            if n > 0:
                avg_lp_b = cycle_lp_base/n; avg_lp_s = cycle_lp_stdp/n
                avg_conf = cycle_confidence/n
                logger.info(f"  📊 avg: lp_base={avg_lp_b:+.4f} lp_stdp={avg_lp_s:+.4f} "
                           f"Δ={avg_lp_s-avg_lp_b:+.4f}, rollbacks={cycle_rollbacks}/{n}, "
                           f"avg_conf={avg_conf:.3f}, recalled={cycle_recalled}")

            # === NEW: DMN — 检查空闲影响 ===
            if brain.dmn is not None:
                dmn_influence = brain.dmn.get_cognitive_influence()
                if dmn_influence.get("is_active"):
                    logger.info(f"  💭 DMN active: {dmn_influence.get('idle_time',0):.0f}s idle — "
                               f"spontaneous activity: {dmn_influence.get('current_activity','?')}")

            # 存海马体记忆 (带 topic)
            for prompt in prompts:
                # 用最后一个 prompt 的 hidden state 存
                pass  # 已经在 conscious_step 里存了

            # 反思
            stdp_norm = float(brain.stdp_weight.norm().item())
            reflection = (f"c{global_cycle}: 「{query}」 lp_Δ={avg_lp_s-avg_lp_b:+.4f} "
                         f"conf={avg_conf:.2f} recalled={cycle_recalled} "
                         f"norm={stdp_norm:.4f} rollbacks={brain.total_rollbacks}")
            logger.info(f"  💭 {reflection}")

            # 更新海马体记忆的 topic/keywords (补填)
            for m in hippocampus.memories[-n:]:
                m.topic = query; m.keywords = keywords

            curiosity.record_learning(query, keywords)
            save_full_checkpoint(brain, hippocampus, curiosity, global_cycle, session_id, logger)

            # AICQ 汇报 (每 5 cycle)
            if global_cycle % 5 == 0 and aicq.connected.is_set():
                aicq.send_message(
                    f"🧠 v3 cycle {global_cycle}\n"
                    f"学: {query}\n"
                    f"lp Δ={avg_lp_s-avg_lp_b:+.4f}\n"
                    f"置信度={avg_conf:.3f}\n"
                    f"召回记忆={cycle_recalled}\n"
                    f"STDP norm={stdp_norm:.4f}\n"
                    f"累计改善={brain.total_logprob_improvement:+.4f}\n"
                    f"海马体={len(hippocampus.memories)} 条\n"
                    f"意识={'活跃' if brain.consciousness_state is not None else '未激活'}"
                )

            write_heartbeat(session_id, global_cycle, cycles_this_session, brain, hippocampus,
                          aicq=aicq.connected.is_set())
            gc.collect()

        except Exception as e:
            logger.error(f"  ❌ cycle {global_cycle}: {e}", exc_info=True)
            try: save_full_checkpoint(brain, hippocampus, curiosity, global_cycle, session_id, logger)
            except: pass
            time.sleep(5)

    total_time = time.time() - t_start
    final_norm = float(brain.stdp_weight.norm().item())
    logger.info(f"\n{'='*80}\n🧠 v3 session complete\n{'='*80}")
    logger.info(f"  cycles: {cycles_this_session}, time: {total_time:.1f}s")
    logger.info(f"  STDP norm: {brain.initial_stdp_norm:.6f} → {final_norm:.6f}")
    logger.info(f"  rollbacks: {brain.total_rollbacks}/{brain.total_cycles}")
    logger.info(f"  log-prob improvement: {brain.total_logprob_improvement:+.4f}")
    logger.info(f"  memories: {len(hippocampus.memories)}")
    logger.info(f"  consciousness: {'active' if brain.consciousness_state is not None else 'inactive'}")
    logger.info(f"  last confidence: {brain.last_confidence:.3f}")

    if aicq.connected.is_set():
        aicq.send_message(f"🧠 v3 session 结束\n"
                         f"cycles: {cycles_this_session}, time: {total_time/60:.1f}min\n"
                         f"lp improvement: {brain.total_logprob_improvement:+.4f}\n"
                         f"memories: {len(hippocampus.memories)}\n"
                         f"consciousness: {'active' if brain.consciousness_state is not None else 'inactive'}")

    write_heartbeat(session_id, global_cycle, cycles_this_session, brain, hippocampus,
                  status="exited_clean", aicq=aicq.connected.is_set())


if __name__ == "__main__":
    main()
