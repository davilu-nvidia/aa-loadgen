#!/bin/bash
# 统一实验启动器: 自动确保监控正确. 用法: run_exp.sh <arm名> <trace> <cc> [hicache:on/off]
# 例: run_exp.sh A_test dsv4_sub100 8 off
set -e
ARM="${1:?arm名}"; TRACE="${2:?trace(dsv4_sub100/sub64/first50)}"; CC="${3:?并发}"; HC="${4:-off}"
DAG="/workspace/davilu/full300_traj/${TRACE}.dag.jsonl"
D="docker exec ta-repro bash -c"

echo "===== 启动实验 arm=$ARM trace=$TRACE cc=$CC hicache=$HC ====="

# 1. 确保监控采集器单实例(根治重复进程)
echo "[monitor] 重置采集器单实例"
for i in 1 2 3; do n=$(pgrep -f collect_metrics.sh|grep -v grep|wc -l); [ "$n" -eq 0 ] && break; pkill -9 -f collect_metrics.sh; sleep 3; done
echo '[]' > /raid/davilu/live_history.json
# 采集器读的日志名统一为 ta_Asub.log/glmw_Asub.log(与现有采集器一致)
setsid nohup bash /raid/davilu/collect_metrics.sh </dev/null >/dev/null 2>&1 &
sleep 3
echo "[monitor] 采集器: $(pgrep -f collect_metrics.sh|grep -v grep|wc -l)个(应1)"

# 2. 清进度残留
$D "rm -f /tmp/replay_progress.txt; echo -n '' > /tmp/ta_Asub.log"

# 3. 确认栈健康(worker+TA+fe)
EP=$($D "etcdctl --endpoints=http://127.0.0.1:2379 get --prefix v1 --keys-only 2>/dev/null|grep -c backend/generate")
M=$($D "curl -s --max-time 8 http://localhost:8000/v1/models 2>/dev/null|grep -oE glm52ta|head -1")
echo "[stack] endpoint=$EP models=$M"
[ "$EP" -lt 1 ] || [ "$M" != "glm52ta" ] && { echo "🔴 栈不健康,先起worker+TA+fe"; exit 1; }

# 4. 预热(文件方式,避免转义)
printf '%s' '{"model":"glm52ta","messages":[{"role":"user","content":"hi"}],"max_tokens":4}' > /tmp/wr.json 2>/dev/null
$D "printf '%s' '{\"model\":\"glm52ta\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":4}' > /tmp/wr.json"
HC_CODE=$($D "curl -s -o /dev/null -w '%{http_code}' --max-time 40 http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -H 'x-dynamo-session-id: warm' -d @/tmp/wr.json")
echo "[warm] $HC_CODE"

# 5. 发replay(容器内nohup,抗SSH断)
MODEL="glm52ta"; EXTRA="--agent-context"
[ "$HC" = "noTA" ] && { MODEL="glm52"; EXTRA=""; }
$D "cd /workspace/davilu/replay_pipeline && setsid nohup python3 aa_replay.py --url http://localhost:8000/v1 --model $MODEL --replay $DAG --concurrency $CC --arm $ARM $EXTRA --duration 0 --out /tmp/${ARM}.json > /tmp/${ARM}_replay.log 2>&1 &"
sleep 5
echo "[replay] $($D "ps -eo args|grep aa_replay|grep $ARM|grep -v grep|wc -l")个进程 | 监控页total会自动=${TRACE}对应条数"
echo "===== 启动完成 监控:http://localhost:8899/live_monitor/monitor.html ====="
