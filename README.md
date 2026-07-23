# aa-loadgen

Multi-turn agentic inference load generator for A/B benchmarking serving
infrastructure (built for NVIDIA Dynamo `thunderagent_router` program-aware
scheduling). One CLI, two modes:

| Mode | What it does | When to use |
|---|---|---|
| `synth` | Synthesizes AA-AgentPerf-style load (controllable ISL/OSL/tool-delay distributions) | Reproducible stress tests; tune ISL to saturate KV cache |
| `replay` | Replays recorded real trajectories (dag_jsonl) verbatim | Faithful A/B on real agent workloads |

Both modes share one measurement core (`aaload/core.py`): identical SSE
parsing, usage-based token counting, and report format.

## Usage

```bash
# Synthetic load (AA-AgentPerf methodology)
python aa_loadgen.py synth --url http://localhost:8000/v1 --model <MODEL> \
    --concurrency 64 --duration 600 --warmup 60 --arm A --out aa_armA.json

# Replay recorded trajectories
python aa_loadgen.py replay --url http://localhost:8000/v1 --model <MODEL> \
    --replay traj.dag.jsonl --concurrency 64 --arm A --agent-context \
    --out armA.json

# Loop the pool with real KV pressure (nonce breaks cross-replica prefix sharing)
python aa_loadgen.py replay --replay traj.dag.jsonl --loop --nonce \
    --duration 600 --concurrency 64 --arm B --out armB.json
```

Requires `aiohttp`. Targets any OpenAI-compatible `/v1/chat/completions`
endpoint.

## Layout

```
aa_loadgen.py        CLI entry (synth / replay subcommands)
aaload/core.py       shared: streaming client, metrics, percentiles, report
aaload/synth.py      synthetic workload (AA distributions)
aaload/replay.py     trajectory replay
replay/              offline tools: recording hook, convert_to_dag.py, examples
monitor/             experiment launcher + live monitoring dashboard
```

## Synth mode (AA-AgentPerf methodology)

Following [Artificial Analysis AA-AgentPerf](https://artificialanalysis.ai/methodology/agentperf):

- **Multi-turn programs**: each virtual agent = one program (session) issuing
  N sequential turns (6-40) with a growing conversation context.
- **Realistic sequence lengths**: per-request ISL 5K-131K, mean ~27K
  (lognormal); variable OSL (short tool-calls vs long reasoning).
- **Simulated tool latency**: inter-turn delay (median 1s, 0.1-5s); tools are
  not actually executed (replay model, matching AA).
- **Program-aware routing support**: injects `x-dynamo-session-id` so a
  program-level scheduler can group turns and pause/resume at tool boundaries.
- **Fair A/B**: per-session unique nonce prefix in the system prompt defeats
  cross-arm KV cache reuse.
- `--ignore-eos` forces exact sampled OSL (vLLM/SGLang extension).

## Replay mode

- Reads dag_jsonl (from `replay/convert_to_dag.py`): one session per line,
  `{session_id, turns:[{messages, max_tokens, model, delay}]}`.
- Closed-loop: keeps `--concurrency` sessions in flight until the pool is
  exhausted (or `--duration` elapses; `--loop` cycles the pool).
- Replays each turn's recorded messages verbatim; `max_tokens` = recorded OSL
  (`--ignore-eos`, default on, reproduces exact decode length); sleeps the
  recorded tool wall-clock between turns.
- `--nonce`: per-replica nonce injected into the **first message content**
  (and the session id). Engine prefix caches key on token sequences, so a
  header-only nonce would not break cross-replica KV sharing.
- Live progress written to `<out>.progress` (override: `--progress-file`).

## Metrics

Aligned with AA: P50/P95 TTFT, P25/median output speed, P50/P95 TPOT,
per-session E2E, steps/min, output token throughput. Token counts prefer the
server-reported `usage.completion_tokens` (requested via
`stream_options.include_usage`; disable with `--no-stream-usage`) over SSE
chunk counting, which under-counts on servers that batch tokens per chunk.
`--warmup N` discards samples from the first N seconds so reports reflect
steady state.

## Why synthetic load

Real agent harnesses (e.g. mini-SWE-agent on SWE-bench) couple "can the model
solve the task" with "how well does the router schedule", produce
non-reproducible trajectories, and require per-instance Docker sandboxes. A
synthetic AA-style generator gives a controllable, reproducible workload whose
ISL can be tuned to saturate the KV cache and trigger scheduler pause/resume —
isolating the routing layer as the only variable in an A/B comparison.

## Changes vs v1 (not directly comparable to old numbers)

- Synth per-request **mean** ISL now matches the 27K target (v1 sampled the
  *peak*, so the realized mean was ~half).
- Output token counts use engine `usage` in both modes (v1 synth counted SSE
  chunks).
- Replay `--nonce` now actually breaks prefix caching (v1 put it only in the
  header).
- Optional `--warmup` discards the cold-start transient.
