# Trajectory Replay Pipeline

Record **real** multi-turn agentic trajectories (mini-SWE-agent solving SWE-bench)
and replay them for faithful, reproducible A/B benchmarking of agentic serving infra
(Dynamo ThunderAgent on/off, KV router, TP topology, H20 vs H100, ...).

Complements the synthetic `aa_loadgen.py`: instead of sampling load from distributions,
this pipeline captures and replays **actual** agent behavior (real prompts, real output
lengths, real inter-turn tool wall-clock).

## Why replay (vs synthetic)

A trajectory is an **input-load recording**, not a performance recording. It fixes:
- `messages` — the exact context sent each turn (real, accumulating)
- `max_tokens` — the real output length per turn (OSL)
- `delay` — real inter-turn tool wall-clock (docker/pytest), NOT LLM latency

Performance (TTFT/throughput) is measured **live at replay time** on the target infra.
=> Same trajectory replays on any hardware/router config for a fair A/B. ISL is
implicit in `messages` and re-tokenized by the target model at replay time (portable
across tokenizers); OSL is pinned via `max_tokens` (+ `ignore_eos` to force exact length).

NOTE: a trajectory captures a *specific model's* behavior. Replaying a DSV4 trajectory
on GLM is valid as a **load test** but does not represent GLM's own agent behavior —
record a fresh trajectory per model for that.

## Pipeline

```
1. RECORD   mini-SWE-agent + <model> on Dynamo/SGLang, solving SWE-bench Lite
            recording_hook.py.snippet appended to minisweagent/models/vllm_model.py,
            activated by MSWEA_RECORD_DIR -> trajectory_turns.jsonl (one line per turn)

2. CONVERT  convert_to_dag.py: trajectory_turns.jsonl -> <model>.dag.jsonl
            groups turns by trajectory_id into sessions; OSL from usage/tokenizer;
            delay = gap(prev last_token_ts -> this request_ts) = tool wall-clock

3. REPLAY   aa_replay.py: closed-loop concurrency replay of the dag_jsonl against
            an OpenAI endpoint. Injects x-dynamo-session-id; ignore_eos pins exact OSL.
            Metrics: P95 TTFT, P25/median output speed, steps/min, throughput.
```

## Usage

```bash
# 1. Record (on the recording host, mini-SWE-agent dir)
export MSWEA_RECORD_DIR=/path/to/rec_traj
export MSWEA_SEND_ROUTER_EXTRA_BODY=0     # record without TA/agent_context
mini-extra swebench --config swebench.yaml --subset lite --split test \
    --slice 0:300 --model <name> -o rec_out -w 128 --redo-existing

# 2. Convert to dag_jsonl (aiperf-compatible format)
python replay/convert_to_dag.py \
    --in  $MSWEA_RECORD_DIR/trajectory_turns.jsonl \
    --out <model>.dag.jsonl \
    --tokenizer /path/to/model --model-name <name>

# 3. Replay for A/B (arm A = TA on, arm B = TA off — swap the --url endpoint)
python replay/aa_replay.py --url http://localhost:8000/v1 --model <name> \
    --replay <model>.dag.jsonl --concurrency 128 --arm A --out armA.json
```

## Files
- `aa_replay.py` — closed-loop dag_jsonl replay engine (companion to ../aa_loadgen.py)
- `convert_to_dag.py` — recorded turns -> dag_jsonl converter
- `recording_hook.py.snippet` — monkeypatch appended to mini-SWE-agent's vllm_model.py
