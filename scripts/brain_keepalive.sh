#!/bin/bash
# brain_keepalive.sh — 最简保活: 死了就重启, 永远跑
# 用法: setsid bash brain_keepalive.sh &
RUNTIME_DIR="/home/z/my-project/brain_runtime"
SCRIPTS_DIR="/home/z/my-project/scripts"
DAEMON="$SCRIPTS_DIR/brain_daemon_v2_fixed.py"
LOG="/tmp/brain_v2_keepalive.log"

echo "[$(date '+%H:%M:%S')] keepalive starting" >> "$LOG"

while true; do
    # 检查是否已有 daemon 在跑
    if pgrep -f "brain_daemon_v2_fixed" > /dev/null; then
        sleep 10
        continue
    fi
    
    echo "[$(date '+%H:%M:%S')] daemon not running, starting..." >> "$LOG"
    cd /home/z/my-project
    python3 "$DAEMON" --max-cycles 50 --master-id 1000008 >> "$LOG" 2>&1 &
    DAEMON_PID=$!
    echo "[$(date '+%H:%M:%S')] started PID=$DAEMON_PID" >> "$LOG"
    
    # 等它结束 (正常退出或崩溃)
    wait $DAEMON_PID
    exit_code=$?
    echo "[$(date '+%H:%M:%S')] daemon exited (code=$exit_code), will restart in 5s" >> "$LOG"
    sleep 5
done
