#!/bin/bash
# brain_supervisor_aicq.sh — AICQ-enabled brain supervisor
# Runs brain_daemon_aicq.py with auto-restart
set -uo pipefail

RUNTIME_DIR="/home/z/my-project/brain_runtime"
SCRIPTS_DIR="/home/z/my-project/scripts"
DAEMON="$SCRIPTS_DIR/brain_daemon_aicq.py"
LOG_DIR="$RUNTIME_DIR/logs"
PID_FILE="$RUNTIME_DIR/supervisor_aicq.pid"
SUP_LOG="$LOG_DIR/supervisor_aicq.log"

mkdir -p "$LOG_DIR"

if [ -f "$PID_FILE" ] && kill -0 "$(cat $PID_FILE)" 2>/dev/null; then
    echo "[!] supervisor already running (PID $(cat $PID_FILE))"
    exit 1
fi
echo $$ > "$PID_FILE"
trap 'log "received signal, shutting down..."; kill $DAEMON_PID 2>/dev/null || true; wait $DAEMON_PID 2>/dev/null || true; rm -f "$PID_FILE"; exit 0' EXIT INT TERM

MAX_RESTARTS=50
BACKOFF_BASE=5
BACKOFF_MAX=300
restart_count=0
session_start=$(date +%s)

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [supervisor-aicq] $*" | tee -a "$SUP_LOG"; }

log "🧠 brain_supervisor_aicq starting (PID=$$)"
log "   daemon: $DAEMON"
log "   master_id: 1000008"
log "   max_restarts: $MAX_RESTARTS"
log "========================================"

while [ $restart_count -lt $MAX_RESTARTS ]; do
    log "▶ starting daemon (attempt $((restart_count+1))/$MAX_RESTARTS)"

    start_time=$(date +%s)
    python3 "$DAEMON" --max-cycles 10 --master-id 1000008 > "$LOG_DIR/daemon_aicq_latest.log" 2>&1 &
    DAEMON_PID=$!
    wait $DAEMON_PID
    exit_code=$?
    end_time=$(date +%s)
    duration=$((end_time - start_time))

    log "◀ daemon exited (code=$exit_code, duration=${duration}s)"

    if [ $exit_code -eq 0 ] && [ $duration -gt 30 ]; then
        restart_count=0
        backoff=$BACKOFF_BASE
        log "  ✅ clean exit after ${duration}s, resetting restart counter"
    else
        restart_count=$((restart_count + 1))
        backoff=$((BACKOFF_BASE * (2 ** (restart_count - 1))))
        if [ $backoff -gt $BACKOFF_MAX ]; then backoff=$BACKOFF_MAX; fi
        log "  ⚠️ crash/short run, backing off ${backoff}s (restart $restart_count/$MAX_RESTARTS)"
    fi

    if [ $restart_count -lt $MAX_RESTARTS ]; then
        sleep $backoff
    fi
done

total_time=$(($(date +%s) - session_start))
log "========================================"
log "🧠 supervisor finished after ${total_time}s"
