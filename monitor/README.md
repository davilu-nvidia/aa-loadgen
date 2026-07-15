# 实时监控栈 (Agentic Serving A/B/C)

ThunderAgent / Dynamo / HiCache replay 实验的实时监控。展示进度、TA pause/resume、
GPU 实测 util vs TA 预估 util、HiCache host offload、TTFT/TPOT/Session-E2E 延迟。

## 组件

| 文件 | 作用 | 运行位置 |
|---|---|---|
| `run_exp.sh` | **一键实验启动器**,自动确保监控正确 | 服务器 |
| `collect_metrics.sh` | 每 5s 采集 metrics → `live_metrics.json` + `live_history.json` | 服务器 |
| `sync.sh` | 每 5s scp 从服务器拉数据到本地 | 本地 |
| `monitor.html` | dashboard 前端(每 5s fetch) | 本地 HTTP :8899 |

## 一键启动实验

```bash
# 用法: run_exp.sh <arm名> <trace> <cc> [on/off/noTA]
bash run_exp.sh A_test dsv4_sub100 32 on
```

自动完成:
1. 采集器单实例(循环杀到 0 再起 1 个 —— 根治重复进程导致的字段丢失)
2. 清 progress / TA 日志残留(避免显示上轮数据)
3. 栈健康检查(endpoint + models,不健康则退出)
4. 文件方式预热(避免 curl 转义 bug)
5. replay 容器内 nohup(抗 SSH 断)
6. total 分母自动(从 trace 名推断: sub100=100 / sub64=230 / first50=50)

## 本地看板

```bash
# 本地拉数据
nohup bash sync.sh &
# 起 HTTP 服务
cd <repo根> && python3 -m http.server 8899 &
# 浏览器打开
open http://localhost:8899/monitor/monitor.html
```

## 采集的指标

- **进度**: sessions done / total, requests ok / err
- **TA 调度**: 累计 pause · resume · still_paused(当前暂停数) · scheduler tick
- **GPU 实测 util vs TA 预估 util**(双线): TA 预估 = Σ活跃program的token_total + buffer_per_program / capacity;
  GPU 实测 = worker 上报的物理 KV 占用。TA 略高(含 buffer 安全垫,预防性预留)。
- **HiCache host offload**: host_used / host_total(有 HiCache 时)
- **延迟 P50**: TTFT(首token) / TPOT(每token ms) / Session E2E(多轮总耗时)

## 已知坑(务必避免)

1. 采集器重复进程 → 旧版覆盖新版字段为 None。必须确认 `pgrep -f collect_metrics.sh | grep -v grep | wc -l` = 1。
2. 查进程用 `ps -eo stat,comm` 而非 `pgrep -f`(后者匹配命令行文本自身,误报)。GPU 真相以 `nvidia-smi --query-compute-apps=pid` 为准。
3. grep 提取 `ttft_p50=0.093` 用 `sed "s/ttft_p50=//"`,别用双 grep(会把 "p50" 的 50 也抓出)。
