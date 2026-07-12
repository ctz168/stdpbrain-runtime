# 🧠 STDP Brain Runtime

Autonomous learning runtime that combines:
- **GPT-2** (from `samaidev/lal` repo's reference model) as the base language model
- **stdpbrain** (`ctz168/stdpbrain` repo) brain modules: cerebellum, basal ganglia, amygdala, locus coeruleus, synaptic plasticity (STDP), etc.
- **AICQ SDK** (`samaidev/AIcqsdk` repo) for messaging the master account
- **Web search + page reader** (z-ai-web-dev-sdk) for autonomous knowledge acquisition

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│              brain_supervisor_aicq.sh                     │
│   (auto-restart wrapper, exponential backoff)             │
├──────────────────────────────────────────────────────────┤
│              brain_daemon_aicq.py                         │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐ │
│  │ Curiosity    │→ │ WebFetcher   │→ │ ContentProcessor│ │
│  │ Engine       │  │ (z-ai CLI)   │  │ (HTML→prompts) │ │
│  └──────────────┘  └──────────────┘  └────────────────┘ │
│         ↑                                       ↓         │
│  ┌──────────────┐                       ┌────────────────┐│
│  │ Hippocampus  │←── stores ──── STDP Brain (768×768)    ││
│  │ (memories)   │              + 9 brain modules (frozen)││
│  └──────────────┘                       └────────────────┘│
│         ↓                                       ↓         │
│  ┌──────────────┐                       ┌────────────────┐│
│  │ ThoughtStream│→ AICQ send ─────────→│  Master 1000008 ││
│  │ (reflections)│   (WebSocket)        │  (you)           ││
│  └──────────────┘                       └────────────────┘│
└──────────────────────────────────────────────────────────┘
```

## Files

- `scripts/brain_daemon_aicq.py` — main daemon: long-running learning loop + AICQ messaging
- `scripts/brain_supervisor_aicq.sh` — auto-restart wrapper
- `scripts/brain_status.sh` — health/progress monitor
- `scripts/bind_master.py` — one-time script to add master as friend
- `scripts/gpt2_stdpbrain_integration.py` — v1: GPT-2 + brain observation (frozen)
- `scripts/gpt2_stdpbrain_v2_train.py` — v2: real STDP training + thought stream
- `scripts/gpt2_stdpbrain_v3_longtrain.py` — v3: 60 wikitext prompts × 2 epochs + external reward
- `scripts/gpt2_stdpbrain_v4_autolearn.py` — v4: web-fetched learning + hippocampus
- `brain_runtime/checkpoints/` — STDP weight snapshots (resumable)
- `brain_runtime/logs/` — daemon + supervisor logs
- `brain_runtime/heartbeat.json` — live status

## Quick Start

```bash
# 1. Configure HF token (for GPT-2 download)
echo "hf_xxx" > brain_runtime/.hf_token
chmod 600 brain_runtime/.hf_token

# 2. One-time: bind brain agent to master account
python3 scripts/bind_master.py --master-id 1000008

# 3. Start the brain container (auto-restart on crash)
nohup bash scripts/brain_supervisor_aicq.sh &

# 4. Monitor
./scripts/brain_status.sh
```

## Trigger Conditions for Brain → Master Messages

The brain will proactively message the master when any of:
- `DA > 0.65` (strong reward signal)
- `|valence| > 0.5` (strong emotion)
- `STDP change rate > 50%` (learning-intensive)
- Every 5 cycles (periodic report)

## Checkpoint & Resume

Every learning cycle saves a checkpoint containing:
- STDP weight matrix (768×768)
- Optimizer state (SGD momentum)
- synaptic_plasticity meta-params (a_plus, a_minus)
- Hippocampus memories
- Curiosity engine state (learned topics/keywords)

On restart, the daemon auto-loads the latest checkpoint and continues.
