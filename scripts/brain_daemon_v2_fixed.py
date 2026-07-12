#!/usr/bin/env python3
"""
brain_daemon_v2_fixed.py — 修复版 brain 守护进程
=================================================

修复 v1 的 3 个关键问题:
1. STDP_W 初始太大 (randn*0.01 → 0.001)
2. 注入 scale 太大 (0.1 → 0.01)
3. 无 log-prob 监控 → 加自动检测+回滚机制

新增:
- M1_monitor: 每 cycle 测当前 prompt 的 log-prob, 与基线对比
- auto_rollback: 如果 log-prob 下降超过阈值, 回滚到上个 checkpoint + 减半 lr
- true_reward: reward = log-prob 改善量 (而非启发式)
- health_check: 每 5 cycle 输出健康报告
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
CKPT_DIR = os.path.join(RUNTIME_DIR, "checkpoints_v2")
LOG_DIR = os.path.join(RUNTIME_DIR, "logs")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

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

# === 修复后的参数 ===
STDP_INIT_STD = 0.001      # 修复1: 初始权重小 10 倍 (norm ~0.77)
STDP_INJECT_SCALE = 0.01   # 修复2: 注入 scale 小 10 倍 (1.3% 扰动)
STDP_LR = 1e-4             # 修复: lr 小 10 倍
STDP_WEIGHT_CLAMP = 0.1    # 修复: clamp 紧 3 倍

CYCLES_PER_SESSION = 10
PROMPTS_PER_CYCLE = 4
MAX_NEW_TOKENS = 8

# M1 监控参数
M1_ROLLBACK_THRESHOLD = -0.05  # log-prob 下降超过这个就回滚
M1_HEALTH_REPORT_INTERVAL = 5  # 每 5 cycle 输出健康报告

# AICQ
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
    logger = logging.getLogger("brain_daemon_v2")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = RotatingFileHandler(os.path.join(LOG_DIR, f"daemon_v2_{session_id}.log"),
                             maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
    fh.setFormatter(fmt); logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    return logger


# ============================================================================
# AICQ Bridge (复用)
# ============================================================================
class AICQBridge:
    def __init__(self, master_id, logger):
        self.master_id = master_id; self.logger = logger
        self.loop = None; self.thread = None
        self.connected = threading.Event()
        self.incoming_messages = []
        self.lock = threading.Lock()
        self.agent_account_id = None

    def start(self):
        self.thread = threading.Thread(target=self._run_loop, daemon=True, name="aicq-loop")
        self.thread.start()
        if not self.connected.wait(timeout=30):
            self.logger.warning("[aicq] failed to connect within 30s")

    def _run_loop(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try: self.loop.run_until_complete(self._start_aicq())
        except Exception as e: self.logger.error(f"[aicq] loop crashed: {e}")

    async def _start_aicq(self):
        from aicq import startLoop
        from aicq.loop import _get_or_create_identity, _loop_ctx

        async def on_message(content, from_id, ctx):
            self.logger.info(f"📥 [aicq] message from {from_id}: {content!r}")
            with self.lock: self.incoming_messages.append((from_id, content))
            return None

        try:
            identity = _get_or_create_identity()
            self.agent_account_id = identity.get("account_id")
            self.logger.info(f"[aicq] agent account_id: {self.agent_account_id}")
        except Exception as e:
            self.logger.warning(f"[aicq] failed to get identity: {e}")

        async def _run_startloop():
            try: await startLoop(on_message, server=AICQ_SERVER)
            except Exception as e: self.logger.error(f"[aicq] startLoop failed: {e}")
            finally: self.connected.clear()

        task = asyncio.create_task(_run_startloop())
        for _ in range(30):
            if _loop_ctx.ws is not None and not _loop_ctx.ws.closed and _loop_ctx.access_token:
                self.connected.set()
                self.logger.info("[aicq] WebSocket connected")
                break
            await asyncio.sleep(0.5)
        else:
            self.logger.warning("[aicq] WS not ready after 15s")
            self.connected.set()
        await task

    def send_message(self, content, to_id=None):
        if not self.loop or not self.loop.is_running():
            self.logger.warning("[aicq] loop not running"); return False
        target = to_id or self.master_id
        future = asyncio.run_coroutine_threadsafe(self._send_async(target, content), self.loop)
        try:
            future.result(timeout=10); return True
        except Exception as e:
            self.logger.error(f"[aicq] send failed: {e}"); return False

    async def _send_async(self, to_id, content):
        from aicq import loop_send_message
        await loop_send_message(to_id, content)
        self.logger.info(f"📤 [aicq] sent to {to_id}: {content[:80]!r}")

    def get_incoming(self):
        with self.lock:
            msgs = self.incoming_messages.copy()
            self.incoming_messages.clear()
        return msgs


# ============================================================================
# Web/Curiosity/Content (复用, 精简)
# ============================================================================
class CuriosityEngine:
    def __init__(self):
        self.learned_topics = []; self.learned_keywords = []; self.cycle_count = 0
    def generate_query(self):
        self.cycle_count += 1
        if random.random() < 0.3 and self.learned_keywords:
            seed = random.choice(self.learned_keywords)
            assoc = {"brain":["consciousness","neurons"], "quantum":["entanglement"], "DNA":["RNA","proteins"], "climate":["oceans","weather"], "memory":["hippocampus","sleep"], "sleep":["dreams","REM"]}
            related = assoc.get(seed, [])
            if related:
                t = random.choice(related); return f"{seed} {t}", f"关联探索: {seed} → {t}"
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
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
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


@dataclass
class Memory:
    id: str; timestamp: float; source_url: str; source_title: str
    topic: str; keywords: List[str]; prompts: List[str]
    reward: float; stdp_changed_pct: float; reflection: str; cycle: int
    logprob_baseline: float; logprob_stdp: float; logprob_delta: float


class HippocampusLite:
    def __init__(self): self.memories = []
    def store(self, mem): self.memories.append(mem)
    def get_summary(self):
        if not self.memories: return {"total": 0}
        return {"total": len(self.memories), "topics": [m.topic for m in self.memories],
                "avg_reward": sum(m.reward for m in self.memories) / len(self.memories)}
    def state_dict(self): return {"memories": [asdict(m) for m in self.memories]}
    def load_state_dict(self, s):
        self.memories = []
        for m in s.get("memories", []): self.memories.append(Memory(**m))


# ============================================================================
# 修复版 STDP Brain — 加 log-prob 监控 + 自动回滚
# ============================================================================
class STDPBrainV2:
    def __init__(self, logger):
        self.logger = logger
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
            except Exception as e: logger.warning(f"  {name}: {e}")

        # 修复1: 小初始权重
        self.stdp_weight = nn.Parameter(
            torch.randn(HIDDEN_SIZE, HIDDEN_SIZE, device=DEVICE) * STDP_INIT_STD, requires_grad=True
        )
        self.optimizer = torch.optim.SGD([self.stdp_weight], lr=STDP_LR, momentum=0.9, weight_decay=1e-4)
        self.initial_lr = STDP_LR
        logger.info(f"  ✅ STDP_W init: std={STDP_INIT_STD}, norm={float(self.stdp_weight.norm().item()):.4f}")
        logger.info(f"  ✅ inject_scale={STDP_INJECT_SCALE}, lr={STDP_LR}, clamp=±{STDP_WEIGHT_CLAMP}")

        # sp 元参数
        sp = self.modules.get("synaptic_plast")
        self.sp_meta_params = {}
        if sp is not None:
            for sub_name, sub_mod in sp.named_modules():
                for attr in ['a_plus', 'a_minus']:
                    if hasattr(sub_mod, attr):
                        self.sp_meta_params[f"{sub_name}.{attr}"] = getattr(sub_mod, attr)

        self.da_level = 0.5; self.ne_level = 0.5; self.valence = 0.0
        self.initial_stdp_norm = float(self.stdp_weight.norm().item())

        # 修复3: log-prob 监控 + 回滚机制
        self.last_ckpt_state = None  # 用于回滚
        self.consecutive_failures = 0
        self.total_rollbacks = 0
        self.total_cycles = 0
        self.total_logprob_improvement = 0.0

    def compute_logprob(self, gpt2, input_ids, use_stdp=True):
        """计算给定 input_ids 的平均 token log-prob."""
        with torch.no_grad():
            out = gpt2(input_ids, output_hidden_states=True, use_cache=False)
            if use_stdp:
                h = out.hidden_states[-1]
                # 修复2: 小 scale 注入
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

    def step_with_monitoring(self, gpt2, tokenizer, prompt, hidden_now, hidden_prev, logger):
        """带 log-prob 监控的单步训练."""
        # 1. 先测基线 log-prob (无 STDP)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        if input_ids.shape[1] < 2:
            return None, 0, 0, 0, False

        logprob_baseline = self.compute_logprob(gpt2, input_ids, use_stdp=False)
        logprob_stdp_before = self.compute_logprob(gpt2, input_ids, use_stdp=True)

        # 2. 保存回滚状态
        self.last_ckpt_state = {
            "stdp_weight": self.stdp_weight.data.clone(),
            "optimizer_state": self.optimizer.state_dict(),
        }

        # 3. brain 闭环 (简化版)
        snap = {}
        with torch.no_grad():
            # 修复2: 小 scale 注入
            stdp_delta = F.linear(hidden_now, self.stdp_weight)
            modified_hidden = hidden_now + stdp_delta * STDP_INJECT_SCALE

            cb = self.modules.get("cerebellar")
            cerebellar_err = 0.0; corrected = hidden_now
            if cb is not None:
                try:
                    r = cb.forward(hidden_now, hidden_prev, hidden_now)
                    err = r.get("prediction_error", None)
                    if isinstance(err, torch.Tensor): cerebellar_err = float(err.norm().item())
                    corrected = r.get("corrected_output", hidden_now)
                    if not isinstance(corrected, torch.Tensor): corrected = hidden_now
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

            # sp ΔW
            sp = self.modules.get("synaptic_plast")
            delta_w = None
            if sp is not None:
                try:
                    new_w = sp.forward(pre_activity=hidden_prev, post_activity=hidden_now,
                                       dopamine_level=da_level, weights=self.stdp_weight.data.clone())
                    delta_w = (new_w - self.stdp_weight.data).detach()
                except: delta_w = None

        # 4. 真正的 reward = log-prob 改善 (而非启发式)
        # 我们希望 STDP 让 log-prob 变高, 所以 reward = (logprob_stdp - logprob_baseline)
        # 但这是"评估 reward", 用于决定更新方向
        true_reward = logprob_stdp_before - logprob_baseline  # 当前 STDP 的效果
        combined_reward = 0.5 * true_reward + 0.5 * (da_level - 0.5)
        snap["true_reward"] = true_reward
        snap["combined_reward"] = combined_reward

        # 5. 更新 STDP_W (伪梯度)
        if delta_w is not None:
            with torch.no_grad():
                # 修复: 只有当 reward 为正时才强化 (沿 delta_w 方向走)
                # reward 为负时, 反向走 (抑制 delta_w)
                pseudo_grad = -delta_w * combined_reward * 0.01  # scale 也小 10 倍
                if self.stdp_weight.grad is None:
                    self.stdp_weight.grad = pseudo_grad.clone()
                else:
                    self.stdp_weight.grad.copy_(pseudo_grad)

                # sp 元更新
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
        snap["stdp_grad_norm"] = float(self.stdp_weight.grad.norm().item()) if self.stdp_weight.grad is not None else 0.0

        if self.stdp_weight.grad is not None:
            gn = self.stdp_weight.grad.norm().item()
            if gn > 10.0: self.stdp_weight.grad.mul_(10.0 / gn)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.stdp_weight.data.clamp_(-STDP_WEIGHT_CLAMP, STDP_WEIGHT_CLAMP)

        # 6. 修复3: 测更新后的 log-prob, 决定是否回滚
        logprob_stdp_after = self.compute_logprob(gpt2, input_ids, use_stdp=True)
        logprob_delta = logprob_stdp_after - logprob_stdp_before  # 这一步 STDP 更新带来的改善

        rolled_back = False
        if logprob_delta < M1_ROLLBACK_THRESHOLD:
            # STDP 更新让 log-prob 下降太多, 回滚!
            logger.warning(f"    ⚠️ M1 回滚: log-prob Δ={logprob_delta:+.4f} < {M1_ROLLBACK_THRESHOLD}")
            self.stdp_weight.data.copy_(self.last_ckpt_state["stdp_weight"])
            self.optimizer.load_state_dict(self.last_ckpt_state["optimizer_state"])
            rolled_back = True
            self.total_rollbacks += 1
            self.consecutive_failures += 1
            # 连续失败 3 次, 减半 lr
            if self.consecutive_failures >= 3:
                new_lr = self.optimizer.param_groups[0]["lr"] * 0.5
                self.optimizer.param_groups[0]["lr"] = new_lr
                logger.warning(f"    📉 连续 {self.consecutive_failures} 次回滚, lr 减半 → {new_lr:.6f}")
                self.consecutive_failures = 0
        else:
            self.consecutive_failures = 0
            self.total_logprob_improvement += logprob_delta

        self.total_cycles += 1
        snap["logprob_baseline"] = logprob_baseline
        snap["logprob_stdp_before"] = logprob_stdp_before
        snap["logprob_stdp_after"] = logprob_stdp_after
        snap["logprob_delta"] = logprob_delta
        snap["rolled_back"] = rolled_back

        return snap, modified_hidden.detach(), logprob_baseline, logprob_stdp_after, rolled_back

    def save_checkpoint(self, path):
        sp_meta = {k: (float(v) if not isinstance(v, torch.Tensor) else float(v.mean().item())) for k, v in self.sp_meta_params.items()}
        torch.save({
            "stdp_weight": self.stdp_weight.data.clone(),
            "optimizer_state": self.optimizer.state_dict(),
            "sp_meta_params": sp_meta,
            "initial_stdp_norm": self.initial_stdp_norm,
            "da_level": self.da_level, "ne_level": self.ne_level, "valence": self.valence,
            "total_cycles": self.total_cycles,
            "total_rollbacks": self.total_rollbacks,
            "total_logprob_improvement": self.total_logprob_improvement,
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
        self.da_level = ckpt.get("da_level", 0.5); self.ne_level = ckpt.get("ne_level", 0.5); self.valence = ckpt.get("valence", 0.0)
        self.total_cycles = ckpt.get("total_cycles", 0)
        self.total_rollbacks = ckpt.get("total_rollbacks", 0)
        self.total_logprob_improvement = ckpt.get("total_logprob_improvement", 0.0)
        return True


# ============================================================================
# Checkpoint
# ============================================================================
def save_full_checkpoint(brain, hippocampus, curiosity, cycle_num, session_id, logger):
    ckpt_path = os.path.join(CKPT_DIR, f"brain_ckpt_{session_id}_c{cycle_num:04d}.pt")
    brain.save_checkpoint(ckpt_path)
    state_path = os.path.join(CKPT_DIR, f"brain_state_{session_id}_c{cycle_num:04d}.json")
    state = {"cycle": cycle_num, "session_id": session_id, "timestamp": time.time(),
             "hippocampus": hippocampus.state_dict(), "curiosity": curiosity.state_dict(),
             "stdp_norm": float(brain.stdp_weight.norm().item()), "initial_stdp_norm": brain.initial_stdp_norm,
             "total_rollbacks": brain.total_rollbacks, "total_cycles": brain.total_cycles,
             "total_logprob_improvement": brain.total_logprob_improvement}
    with open(state_path, 'w', encoding='utf-8') as f: json.dump(state, f, ensure_ascii=False, indent=2, default=str)
    with open(os.path.join(CKPT_DIR, "LATEST"), 'w') as f: f.write(f"{session_id}:{cycle_num:04d}\n")
    logger.info(f"  💾 checkpoint: cycle={cycle_num} stdp_norm={state['stdp_norm']:.6f} rollbacks={brain.total_rollbacks}")


def load_latest_checkpoint(brain, hippocampus, curiosity, logger):
    latest_path = os.path.join(CKPT_DIR, "LATEST")
    if not os.path.exists(latest_path):
        logger.info("  [*] no previous checkpoint, starting fresh"); return 0, None
    with open(latest_path) as f: line = f.read().strip()
    parts = line.split(":")
    if len(parts) != 2: return 0, None
    prev_session, prev_cycle_str = parts; prev_cycle = int(prev_cycle_str)
    ckpt_path = os.path.join(CKPT_DIR, f"brain_ckpt_{prev_session}_c{prev_cycle:04d}.pt")
    state_path = os.path.join(CKPT_DIR, f"brain_state_{prev_session}_c{prev_cycle:04d}.json")
    if not os.path.exists(ckpt_path) or not os.path.exists(state_path): return 0, None
    try:
        brain.load_checkpoint(ckpt_path)
        with open(state_path) as f: state = json.load(f)
        hippocampus.load_state_dict(state.get("hippocampus", {}))
        curiosity.load_state_dict(state.get("curiosity", {}))
        logger.info(f"  ✅ resumed from session={prev_session} cycle={prev_cycle} "
                    f"(stdp_norm={state.get('stdp_norm', 0):.6f}, rollbacks={state.get('total_rollbacks', 0)})")
        return prev_cycle, prev_session
    except Exception as e:
        logger.warning(f"  [!] load failed: {e}, starting fresh"); return 0, None


def write_heartbeat(session_id, cycle, total_cycles, brain, hippocampus, status="running", aicq_connected=False):
    hb = {"session_id": session_id, "cycle": cycle, "total_cycles_this_session": total_cycles,
          "status": status, "timestamp": time.time(),
          "stdp_norm": float(brain.stdp_weight.norm().item()), "initial_stdp_norm": brain.initial_stdp_norm,
          "stdp_delta_pct": (float(brain.stdp_weight.norm().item()) - brain.initial_stdp_norm) / max(brain.initial_stdp_norm, 1e-8) * 100,
          "memories": len(hippocampus.memories),
          "da": brain.da_level, "ne": brain.ne_level, "valence": brain.valence,
          "pid": os.getpid(), "aicq_connected": aicq_connected,
          "total_rollbacks": brain.total_rollbacks, "total_cycles": brain.total_cycles,
          "total_logprob_improvement": brain.total_logprob_improvement,
          "current_lr": brain.optimizer.param_groups[0]["lr"]}
    with open(os.path.join(RUNTIME_DIR, "heartbeat.json"), 'w') as f: json.dump(hb, f, indent=2)


class GracefulExit:
    def __init__(self):
        self.should_exit = False
        signal.signal(signal.SIGTERM, self._handler)
        signal.signal(signal.SIGINT, self._handler)
    def _handler(self, signum, frame):
        logging.getLogger("brain_daemon_v2").info(f"  [signal] received {signum}")
        self.should_exit = True


# ============================================================================
# 主循环
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cycles", type=int, default=CYCLES_PER_SESSION)
    parser.add_argument("--master-id", type=str, default=MASTER_ID_DEFAULT)
    args = parser.parse_args()

    session_id = time.strftime("%Y%m%d_%H%M%S")
    logger = setup_logging(session_id)

    logger.info("=" * 80)
    logger.info("🧠 brain_daemon_v2_fixed starting (with log-prob monitoring + auto-rollback)")
    logger.info("=" * 80)
    logger.info(f"  session_id: {session_id}, master_id: {args.master_id}")
    logger.info(f"  FIXES: init_std={STDP_INIT_STD}, inject_scale={STDP_INJECT_SCALE}, lr={STDP_LR}, clamp=±{STDP_WEIGHT_CLAMP}")
    logger.info(f"  M1 rollback threshold: {M1_ROLLBACK_THRESHOLD}")

    aicq = AICQBridge(args.master_id, logger)
    aicq.start()
    if aicq.connected.is_set():
        logger.info(f"  ✅ AICQ connected, agent_id={aicq.agent_account_id}")
        aicq.send_message(
            f"🧠 brain v2 (fixed) 上线\n"
            f"修复: 小初始权重 + 小注入scale + log-prob监控+自动回滚\n"
            f"现在每步都会检测 log-prob 是否改善, 下降超过 {M1_ROLLBACK_THRESHOLD} 会自动回滚."
        )

    logger.info("  loading GPT-2...")
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2 = GPT2LMHeadModel.from_pretrained("gpt2").eval().to(DEVICE)
    for p in gpt2.parameters(): p.requires_grad_(False)
    logger.info(f"  ✅ GPT-2 loaded ({sum(p.numel() for p in gpt2.parameters())/1e6:.1f}M)")

    brain = STDPBrainV2(logger)
    curiosity = CuriosityEngine()
    fetcher = WebFetcher()
    hippocampus = HippocampusLite()

    prev_cycle, _ = load_latest_checkpoint(brain, hippocampus, curiosity, logger)
    global_cycle = prev_cycle

    graceful = GracefulExit()
    logger.info(f"  starting at global_cycle={global_cycle}, initial_stdp_norm={brain.initial_stdp_norm:.6f}")
    logger.info("=" * 80)

    t_session_start = time.time()
    cycles_this_session = 0

    while not graceful.should_exit:
        if args.max_cycles > 0 and cycles_this_session >= args.max_cycles:
            logger.info(f"  reached max_cycles={args.max_cycles}, exiting cleanly"); break

        global_cycle += 1; cycles_this_session += 1
        logger.info(f"\n{'─' * 80}")
        logger.info(f"📚 cycle {global_cycle} (session cycle {cycles_this_session})")
        logger.info(f"{'─' * 80}")

        # 检查主人消息
        incoming = aicq.get_incoming()
        for from_id, content in incoming:
            logger.info(f"  📨 from {from_id}: {content!r}")
            if content.lower().strip() in ["stop", "停", "exit", "quit"]:
                graceful.should_exit = True; aicq.send_message("好的, 停止学习并保存."); break
            aicq.send_message(f"收到「{content}」. 当前 cycle {global_cycle}, STDP norm={float(brain.stdp_weight.norm().item()):.4f}, "
                            f"累计 log-prob 改善={brain.total_logprob_improvement:+.4f}, 回滚次数={brain.total_rollbacks}")
        if graceful.should_exit: break

        try:
            query, reason = curiosity.generate_query()
            logger.info(f"  💭 {reason}, query: {query!r}")

            results = fetcher.search(query, num=5)
            if not results: continue
            best_url = None; best_title = ""
            for r in results:
                host = r.get('host_name', '')
                if any(d in host for d in ['wikipedia','britannica','nature','sciencedaily','ncbi','nasa']):
                    best_url = r.get('url'); best_title = r.get('name', ''); break
            if not best_url: best_url = results[0].get('url'); best_title = results[0].get('name', '')

            page = fetcher.read_page(best_url)
            if not page: continue
            prompts = ContentProcessor.extract_prompts(page['text'])
            keywords = ContentProcessor.extract_keywords(page['text'])
            logger.info(f"  📄 {len(page['text'])} chars, {len(prompts)} prompts, kw: {keywords[:3]}")

            # 训练每个 prompt (带监控)
            cycle_logprob_baseline = 0; cycle_logprob_stdp = 0; cycle_rollbacks = 0
            for i, prompt in enumerate(prompts):
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
                with torch.no_grad():
                    out = gpt2(input_ids, output_hidden_states=True, use_cache=False)
                    hidden_now = out.hidden_states[-1][:, -1, :].squeeze(0).detach()
                    if hidden_now.dim() == 1: hidden_now = hidden_now.unsqueeze(0)
                    hidden_prev = out.hidden_states[-1][:, -2, :].squeeze(0).detach() if input_ids.shape[1] > 1 else hidden_now.clone()
                    if hidden_prev.dim() == 1: hidden_prev = hidden_prev.unsqueeze(0)

                snap, modified_hidden, lp_base, lp_stdp, rolled_back = brain.step_with_monitoring(
                    gpt2, tokenizer, prompt, hidden_now, hidden_prev, logger
                )
                if snap is None: continue
                cycle_logprob_baseline += lp_base; cycle_logprob_stdp += lp_stdp
                if rolled_back: cycle_rollbacks += 1
                logger.info(f"     [{i+1}] lp_base={lp_base:+.3f} lp_stdp={lp_stdp:+.3f} "
                            f"Δ={lp_stdp-lp_base:+.4f} {'↩️ rolled back' if rolled_back else '✅ kept'}")

            n_prompts = len(prompts)
            if n_prompts > 0:
                avg_lp_base = cycle_logprob_baseline / n_prompts
                avg_lp_stdp = cycle_logprob_stdp / n_prompts
                logger.info(f"  📊 avg log-prob: baseline={avg_lp_base:+.4f} stdp={avg_lp_stdp:+.4f} "
                            f"Δ={avg_lp_stdp-avg_lp_base:+.4f}, rollbacks={cycle_rollbacks}/{n_prompts}")

            # 反思
            stdp_norm_now = float(brain.stdp_weight.norm().item())
            stdp_delta_pct = (stdp_norm_now - brain.initial_stdp_norm) / max(brain.initial_stdp_norm, 1e-8) * 100
            reflection = (f"cycle {global_cycle}: 学了「{query}」, "
                         f"log-prob Δ={avg_lp_stdp-avg_lp_base:+.4f}, "
                         f"rollbacks={cycle_rollbacks}/{n_prompts}, "
                         f"STDP norm={stdp_norm_now:.4f} ({stdp_delta_pct:+.3f}%), "
                         f"累计改善={brain.total_logprob_improvement:+.4f}")
            logger.info(f"  💭 {reflection}")

            # 存海马体
            mem = Memory(
                id=f"mem_{global_cycle:04d}_{int(time.time())}",
                timestamp=time.time(), source_url=best_url, source_title=best_title,
                topic=query, keywords=keywords, prompts=prompts,
                reward=avg_lp_stdp - avg_lp_base, stdp_changed_pct=0,
                reflection=reflection, cycle=global_cycle,
                logprob_baseline=avg_lp_base, logprob_stdp=avg_lp_stdp,
                logprob_delta=avg_lp_stdp - avg_lp_base,
            )
            hippocampus.store(mem)
            curiosity.record_learning(query, keywords)

            save_full_checkpoint(brain, hippocampus, curiosity, global_cycle, session_id, logger)

            # 健康报告 (每 5 cycle)
            if global_cycle % M1_HEALTH_REPORT_INTERVAL == 0:
                health = (
                    f"🧠 健康报告 (cycle {global_cycle})\n"
                    f"STDP norm: {brain.initial_stdp_norm:.4f} → {stdp_norm_now:.4f} ({stdp_delta_pct:+.3f}%)\n"
                    f"累计 log-prob 改善: {brain.total_logprob_improvement:+.4f}\n"
                    f"总回滚次数: {brain.total_rollbacks} / {brain.total_cycles} 步\n"
                    f"当前 lr: {brain.optimizer.param_groups[0]['lr']:.6f}\n"
                    f"海马体记忆: {len(hippocampus.memories)} 条\n"
                    f"刚学: {query}"
                )
                logger.info(f"  📊 {health}")
                if aicq.connected.is_set() and (global_cycle % 10 == 0 or brain.total_rollbacks > 0):
                    aicq.send_message(health)

            write_heartbeat(session_id, global_cycle, cycles_this_session, brain, hippocampus,
                          aicq_connected=aicq.connected.is_set())
            gc.collect()

        except Exception as e:
            logger.error(f"  ❌ cycle {global_cycle} failed: {e}", exc_info=True)
            try: save_full_checkpoint(brain, hippocampus, curiosity, global_cycle, session_id, logger)
            except: pass
            time.sleep(5)

    total_time = time.time() - t_session_start
    final_norm = float(brain.stdp_weight.norm().item())
    logger.info("\n" + "=" * 80)
    logger.info("🧠 session complete")
    logger.info("=" * 80)
    logger.info(f"  cycles: {cycles_this_session}, time: {total_time:.1f}s")
    logger.info(f"  STDP norm: {brain.initial_stdp_norm:.6f} → {final_norm:.6f} "
                f"({(final_norm-brain.initial_stdp_norm)/max(brain.initial_stdp_norm,1e-8)*100:+.4f}%)")
    logger.info(f"  total rollbacks: {brain.total_rollbacks} / {brain.total_cycles}")
    logger.info(f"  total log-prob improvement: {brain.total_logprob_improvement:+.4f}")
    logger.info(f"  hippocampus memories: {len(hippocampus.memories)}")

    if aicq.connected.is_set():
        aicq.send_message(
            f"🧠 v2 session 结束\n"
            f"cycles: {cycles_this_session}, time: {total_time/60:.1f}min\n"
            f"STDP norm: {final_norm:.4f} (Δ={((final_norm-brain.initial_stdp_norm)/max(brain.initial_stdp_norm,1e-8)*100):+.3f}%)\n"
            f"累计 log-prob 改善: {brain.total_logprob_improvement:+.4f}\n"
            f"回滚: {brain.total_rollbacks}/{brain.total_cycles} 步\n"
            f"记忆: {len(hippocampus.memories)} 条"
        )

    write_heartbeat(session_id, global_cycle, cycles_this_session, brain, hippocampus,
                  status="exited_clean", aicq_connected=aicq.connected.is_set())


if __name__ == "__main__":
    main()
