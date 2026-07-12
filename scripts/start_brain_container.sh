#!/bin/bash
# start_brain_container.sh — 用 setsid 真正脱离会话的启动器
# 用法: ./start_brain_container.sh
#   会用 setsid 启动 supervisor, 完全脱离当前 shell 会话

set -e
RUNTIME_DIR="/home/z/my-project/brain_runtime"
SCRIPTS_DIR="/home/z/my-project/scripts"

# 清理旧的
pkill -f "brain_daemon_aicq" 2>/dev/null || true
pkill -f "brain_supervisor_aicq" 2>/dev/null || true
sleep 1
rm -f "$RUNTIME_DIR/supervisor_aicq.pid"

echo "🧠 starting brain container (fully detached via setsid)..."

# setsid 创建新会话, disown 移出 job 表, nohup 忽略 SIGHUP
# 三重保险确保进程不被会话清理杀死
setsid bash -c "
exec nohup bash $SCRIPTS_DIR/brain_supervisor_aicq.sh > /tmp/brain_aicq_container.log 2>&1
" < /dev/null > /dev/null 2>&1 &

SUP_PID=$!
disown $SUP_PID 2>/dev/null || true

echo "✅ supervisor launched (launcher PID=$SUP_PID)"
sleep 5

# 验证 supervisor 和 daemon 都在运行
SUP_RUNNING=$(pgrep -f "brain_supervisor_aicq" | head -1)
DAEMON_RUNNING=$(pgrep -f "brain_daemon_aicq" | head -1)

if [ -n "$SUP_RUNNING" ]; then
    echo "   ✅ supervisor running: PID=$SUP_RUNNING"
else
    echo "   ❌ supervisor NOT running"
fi

if [ -n "$DAEMON_RUNNING" ]; then
    echo "   ✅ daemon running: PID=$DAEMON_RUNNING"
else
    echo "   ⚠️  daemon not yet started (may be loading GPT-2)"
fi

echo ""
echo "监控: watch -n 5 $SCRIPTS_DIR/brain_status.sh"
echo "日志: tail -f $RUNTIME_DIR/logs/daemon_aicq_latest.log"
echo "停止: pkill -f brain_supervisor_aicq; pkill -f brain_daemon_aicq"
