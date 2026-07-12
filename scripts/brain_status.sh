#!/bin/bash
# brain_status.sh — 查看 brain 容器状态 (支持 AICQ)
RUNTIME_DIR="/home/z/my-project/brain_runtime"
HB="$RUNTIME_DIR/heartbeat.json"
SUP_PID="$RUNTIME_DIR/supervisor_aicq.pid"
SUP_PID_OLD="$RUNTIME_DIR/supervisor.pid"
SUP_LOG="$RUNTIME_DIR/logs/supervisor_aicq.log"
LATEST_LOG="$RUNTIME_DIR/logs/daemon_aicq_latest.log"
CKPT_DIR="$RUNTIME_DIR/checkpoints"

echo "🧠 STDP Brain 容器状态 (AICQ-enabled)"
echo "========================================"

# 进程
SUP_ID=""; [ -f "$SUP_PID" ] && SUP_ID=$(cat "$SUP_PID")
if [ -n "$SUP_ID" ] && kill -0 "$SUP_ID" 2>/dev/null; then
    ELAPSED=$(ps -p $SUP_ID -o etime= | tr -d ' ')
    RSS=$(ps -p $SUP_ID -o rss= | tr -d ' ')
    echo "✅ supervisor: RUNNING (PID=$SUP_ID, elapsed=$ELAPSED, RSS=${RSS}KB)"
else
    echo "❌ supervisor: NOT RUNNING"
fi

DAEMON_PID=$(pgrep -f "brain_daemon_aicq.py" | head -1 || true)
if [ -n "$DAEMON_PID" ]; then
    ELAPSED=$(ps -p $DAEMON_PID -o etime= | tr -d ' ')
    RSS=$(ps -p $DAEMON_PID -o rss= | tr -d ' ')
    echo "✅ daemon: RUNNING (PID=$DAEMON_PID, elapsed=$ELAPSED, RSS=${RSS}KB)"
else
    echo "⚠️  daemon: not in active cycle"
fi
echo ""

# 心跳
if [ -f "$HB" ]; then
    echo "💓 heartbeat:"
    python3 -c "
import json, time
with open('$HB') as f: h = json.load(f)
age = time.time() - h.get('timestamp', 0)
print(f'   session: {h.get(\"session_id\",\"?\")}')
print(f'   global cycle: {h.get(\"cycle\",0)}')
print(f'   status: {h.get(\"status\",\"?\")}')
print(f'   heartbeat age: {age:.0f}s ago' + (' ⚠️ STALE!' if age > 300 else ' ✅ fresh'))
print(f'   STDP norm: {h.get(\"initial_stdp_norm\",0):.6f} → {h.get(\"stdp_norm\",0):.6f} ({h.get(\"stdp_delta_pct\",0):+.4f}%)')
print(f'   hippocampus memories: {h.get(\"memories\",0)}')
print(f'   DA={h.get(\"da\",0):.3f}  NE={h.get(\"ne\",0):.3f}  valence={h.get(\"valence\",0):+.3f}')
aicq = h.get('aicq_connected', False)
print(f'   AICQ: {\"✅ connected\" if aicq else \"❌ disconnected\"}')
" 2>/dev/null || cat "$HB"
else
    echo "❌ no heartbeat yet"
fi
echo ""

# Checkpoints
CKPT_COUNT=$(ls "$CKPT_DIR"/brain_ckpt_*.pt 2>/dev/null | wc -l)
LATEST_CKPT=$(ls -t "$CKPT_DIR"/brain_ckpt_*.pt 2>/dev/null | head -1)
if [ -n "$LATEST_CKPT" ]; then
    CKPT_SIZE=$(du -h "$LATEST_CKPT" | cut -f1)
    CKPT_AGE=$(( $(date +%s) - $(stat -c %Y "$LATEST_CKPT") ))
    echo "💾 checkpoints: $CKPT_COUNT files (latest: $(basename $LATEST_CKPT), $CKPT_SIZE, ${CKPT_AGE}s ago)"
else
    echo "💾 checkpoints: 0"
fi
echo ""

# AICQ identity
IDENTITY_FILE="$RUNTIME_DIR/.aicq-sdk/loop/identity.json"
if [ -f "$IDENTITY_FILE" ]; then
    echo "🔑 AICQ identity:"
    python3 -c "
import json
with open('$IDENTITY_FILE') as f: d = json.load(f)
print(f'   account_id: {d.get(\"account_id\",\"?\")}')
print(f'   signing_pub: {d.get(\"signing_pub\",\"?\")[:32]}...')
" 2>/dev/null
else
    echo "🔑 AICQ identity: not created yet (will be created on first run)"
fi
echo ""

# 日志
if [ -f "$SUP_LOG" ]; then
    echo "📝 supervisor log (last 5):"
    tail -5 "$SUP_LOG" | sed 's/^/   /'
    echo ""
fi

echo "========================================"
echo "监控命令:"
echo "  实时状态:  watch -n 5 $0"
echo "  实时日志:  tail -f $LATEST_LOG"
echo "  停止:      kill \$(cat $SUP_PID 2>/dev/null || cat $SUP_PID_OLD 2>/dev/null)"
