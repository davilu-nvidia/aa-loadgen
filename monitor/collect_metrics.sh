#!/bin/bash
OUT=/raid/davilu/live_metrics.json
HIST=/raid/davilu/live_history.json
[ -f "$HIST" ] || echo '[]' > "$HIST"
while true; do
  EP=$(docker exec ta-repro bash -c "etcdctl --endpoints=http://127.0.0.1:2379 get --prefix v1 --keys-only 2>/dev/null|grep -c backend/generate" 2>/dev/null)
  STAGE=$(docker exec ta-repro bash -c "grep -aoE 'loading shards: [0-9]+%|Capture cuda graph|Init torch|server is fired' /tmp/glmw_new.log 2>/dev/null|tail -1" 2>/dev/null)
  REPLAY=$(docker exec ta-repro bash -c "pgrep -f 'aa_replay.*A_sub64'|grep -v grep|wc -l" 2>/dev/null)
  if [ "${REPLAY:-0}" -ge 1 ]; then
    PROG=$(docker exec ta-repro bash -c "cat /tmp/replay_progress.txt 2>/dev/null" 2>/dev/null); PHASE="running"
  elif [ "${EP:-0}" -ge 1 ]; then PROG="worker就绪,replay即将开始"; PHASE="warming"
  else PROG="worker启动中: ${STAGE:-加载权重}"; PHASE="starting"; fi
  TICK=$(docker exec ta-repro bash -c "strings /tmp/ta_Asub.log 2>/dev/null|grep -c scheduler.tick" 2>/dev/null)
  RESUME=$(docker exec ta-repro bash -c "strings /tmp/ta_Asub.log 2>/dev/null|grep -oE 'resumed=[0-9]+'|grep -oE '[0-9]+'|awk '{s+=\$1}END{print s+0}'" 2>/dev/null)
  STILLP=$(docker exec ta-repro bash -c "strings /tmp/ta_Asub.log 2>/dev/null|grep -oE 'still_paused=[0-9]+'|tail -1|grep -oE '[0-9]+'" 2>/dev/null)
  PAUSEDSUM=$(docker exec ta-repro bash -c "strings /tmp/ta_Asub.log 2>/dev/null|grep -oE 'paused=[0-9]+ marked'|grep -oE '[0-9]+'|awk '{s+=\$1}END{print s+0}'" 2>/dev/null)
  # TA最新预估util(触发前值,util=X ->的X)
  TAUTIL=$(docker exec ta-repro bash -c "strings /tmp/ta_Asub.log 2>/dev/null|grep -oE 'scheduler.util.*util=[0-9.]+'|tail -1|grep -oE 'util=[0-9.]+'|grep -oE '[0-9.]+'" 2>/dev/null)
  GPU=$(docker exec ta-repro bash -c "tail -c 65536 /tmp/glmw_new.log 2>/dev/null|grep -aoE 'token usage: [0-9.]+'|tail -1|grep -oE '[0-9.]+$'" 2>/dev/null)
  RUN=$(docker exec ta-repro bash -c "tail -c 65536 /tmp/glmw_new.log 2>/dev/null|grep -aoE '#running-req: [0-9]+'|tail -1|grep -oE '[0-9]+$'" 2>/dev/null)
  HOST=$(docker exec ta-repro bash -c "curl -s --max-time 3 http://localhost:8091/metrics 2>/dev/null|grep -E 'hicache_host_used_tokens\{'|grep -oE '[0-9.]+$'" 2>/dev/null)
  LASTUTIL=$(docker exec ta-repro bash -c "strings /tmp/ta_Asub.log 2>/dev/null|grep -oE 'util=[0-9.]+ -> [0-9.]+'|tail -1" 2>/dev/null)
  # 实验元信息(从replay命令行+worker日志解析)
  RCMD=$(docker exec ta-repro bash -c "pgrep -af aa_replay|grep -v grep|head -1" 2>/dev/null)
  EXPARM=$(echo "$RCMD"|grep -oE '\-\-arm [A-Za-z0-9_]+'|awk '{print $2}')
  EXPCC=$(echo "$RCMD"|grep -oE '\-\-concurrency [0-9]+'|awk '{print $2}')
  EXPTRACE=$(echo "$RCMD"|grep -oE 'dsv4_[a-z0-9]+'|head -1)
  EXPHC=$(echo "$RCMD"|grep -q glm52ta && echo "过TA" || echo "直连")
  EXPKV=$(docker exec ta-repro bash -c "grep -aoE 'max_total_num_tokens=[0-9]+' /tmp/glmw_new.log 2>/dev/null|tail -1|grep -oE '[0-9]+'" 2>/dev/null)
  TTFT_P50=$(echo "$PROG"|grep -oE "ttft_p50=[0-9.]+"|sed "s/ttft_p50=//")
  TPOT_P50=$(echo "$PROG"|grep -oE "tpot_p50=[0-9.]+"|sed "s/tpot_p50=//")
  E2E_P50=$(echo "$PROG"|grep -oE "e2e_p50=[0-9.]+"|sed "s/e2e_p50=//")
  case "$EXPTRACE" in
    *sub100*) TOTAL=100;; *sub64*) TOTAL=230;; *first50*) TOTAL=50;; *) TOTAL=0;;
  esac
  TS=$(date '+%H:%M:%S')
  cat > $OUT <<EOF
{"ts":"$TS","phase":"$PHASE","progress":"$PROG","tick":"${TICK:-0}","paused_sum":"${PAUSEDSUM:-0}","resume":"${RESUME:-0}","still_paused":"${STILLP:-0}","gpu_util":"${GPU:-0}","ta_util":"${TAUTIL:-0}","running":"${RUN:-0}","host_used":"${HOST:-0}","last_util":"$LASTUTIL","arm":"${EXPARM:-?}","cc":"${EXPCC:-?}","trace":"${EXPTRACE:-?}","kv":"${EXPKV:-?}","route":"$EXPHC","ttft_p50":"${TTFT_P50:-0}","tpot_p50":"${TPOT_P50:-0}","e2e_p50":"${E2E_P50:-0}","total":"${TOTAL:-0}"}
EOF
  python3 - "$HIST" "$TS" "${GPU:-0}" "${RUN:-0}" "${PAUSEDSUM:-0}" "${HOST:-0}" "${RESUME:-0}" "${STILLP:-0}" "${TAUTIL:-0}" "${TTFT_P50:-0}" "${TPOT_P50:-0}" "${E2E_P50:-0}" <<'PYEOF'
import json,sys
h,ts,gpu,run,psum,host,res,sp,tau=sys.argv[1:10]
try: arr=json.load(open(h))
except: arr=[]
arr.append({"ts":ts,"gpu":float(gpu or 0),"run":int(run or 0),"psum":int(psum or 0),"host":float(host or 0),"resume":int(res or 0),"sp":int(sp or 0),"tau":float(tau or 0),"ttft":float(sys.argv[10] or 0),"tpot":float(sys.argv[11] or 0),"e2e":float(sys.argv[12] or 0)})
arr=arr[-80:]
json.dump(arr,open(h,'w'))
PYEOF
  sleep 5
done
