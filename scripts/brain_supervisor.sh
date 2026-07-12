#!/bin/bash
# brain_supervisor.sh — 自动重启的 brain 守护进程
# ============================================
# 功能:
#   - 启动 brain_daemon.py
#   - 进程退出后自动重启 (带退避)
#   - 限制最大重启次数 (防崩溃循环)
#   - 记录 supervisor 日志
#   - SIGTERM/SIGINT 时优雅停止整个系统
#
# 用法:
#   ./brain_supervisor.sh                 # 前台运行
#   nohup ./brain_supervisor.sh &          # 后台运行
#   kill $(cat brain_runtime/supervisor.pid)  # 停止

set -uo pipefail  # no -e: we handle errors manually

RUNTIME_DIR="/home/z/my-project/brain_runtime"
SCRIPTS_DIR="/home/z/my-project/scripts"
DAEMON="$SCRIPTS_DIR/brain_daemon.py"
LOG_DIR="$RUNTIME_DIR/logs"
PID_FILE="$RUNTIME_DIR/supervisor.pid"
SUP_LOG="$LOG_DIR/supervisor.log"

mkdir -p "$LOG_DIR"

# 防止多个 supervisor 同时运行
if [ -f "$PID_FILE" ] && kill -0 "$(cat $PID_FILE)" 2>/dev/null; then
    echo "[!] supervisor already running (PID $(cat $PID_FILE))"
    exit 1
fi
echo $$ > "$PID_FILE"
trap 'log "received signal, shutting down..."; kill $DAEMON_PID 2>/dev/null || true; wait $DAEMON_PID 2>/dev/null || true; rm -f "$PID_FILE"; exit 0' EXIT INT TERM

# 配置
MAX_RESTARTS=50           # 最多重启 50 次 (每次 10 cycles = 500 cycles 总学习量)
BACKOFF_BASE=5            # 基础退避 5 秒
BACKOFF_MAX=300           # 最大退避 5 分钟

restart_count=0
session_start=$(date +%s)

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [supervisor] $*" | tee -a "$SUP_LOG"
}

log "🧠 brain_supervisor starting (PID=$$)"
log "   runtime: $RUNTIME_DIR"
log "   daemon: $DAEMON"
log "   max_restarts: $MAX_RESTARTS"
log "   backoff: ${BACKOFF_BASE}s .. ${BACKOFF_MAX}s"
log "========================================"

while [ $restart_count -lt $MAX_RESTARTS ]; do
    log "▶ starting daemon (attempt $((restart_count+1))/$MAX_RESTARTS)"

    start_time=$(date +%s)
    # Run daemon, capture exit code separately from tee (pipefail-safe)
    python3 "$DAEMON" --max-cycles 10 > "$LOG_DIR/daemon_latest.log" 2>&1 &
    DAEMON_PID=$!
    wait $DAEMON_PID
    exit_code=$?
    end_time=$(date +%s)
    duration=$((end_time - start_time))

    log "◀ daemon exited (code=$exit_code, duration=${duration}s)"

    # 如果 daemon 正常退出 (exit_code=0) 且运行了足够久, 重置退避
    if [ $exit_code -eq 0 ] && [ $duration -gt 30 ]; then
        restart_count=0
        backoff=$BACKOFF_BASE
        log "  ✅ clean exit after ${duration}s, resetting restart counter"
    else
        restart_count=$((restart_count + 1))
        # 指数退避
        backoff=$((BACKOFF_BASE * (2 ** (restart_count - 1))))
        if [ $backoff -gt $BACKOFF_MAX ]; then
            backoff=$BACKOFF_MAX
        fi
        log "  ⚠️ crash/short run, backing off ${backoff}s (restart $restart_count/$MAX_RESTARTS)"
    fi

    if [ $restart_count -lt $MAX_RESTARTS ]; then
        sleep $backoff
    fi
done

total_time=$(($(date +%s) - session_start))
log "========================================"
log "🧠 supervisor finished after ${total_time}s"
log "   total restarts: $restart_count"
log "   check checkpoints: $RUNTIME_DIR/checkpoints/"
log "   check heartbeat: $RUNTIME_DIR/heartbeat.json"
