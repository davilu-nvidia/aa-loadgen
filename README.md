# aa-loadgen

An **AA-AgentPerf-style load generator** for benchmarking multi-turn agentic
inference serving (built for testing NVIDIA Dynamo's `thunderagent_router`
program-aware scheduling).

Following the [Artificial Analysis AA-AgentPerf](https://artificialanalysis.ai/methodology/agentperf)
methodology, it synthesizes realistic agentic coding load instead of single-shot prompts:

- **Multi-turn programs**: each virtual agent = one program (session) issuing N
  sequential turns with a growing conversation context.
- **Realistic sequence lengths**: ISL 5K-131K, mean ~27K (lognormal); variable OSL
  (short tool-calls vs long reasoning).
- **Simulated tool latency**: inter-turn delay (median 1s, 0.1-5s); tools are not
  actually executed (replay model, matching AA).
- **Program-aware routing support**: injects `x-dynamo-session-id` so a program-level
  scheduler can group turns and pause/resume at tool boundaries.
- **Fair A/B**: per-session unique nonce prefix defeats cross-arm KV cache reuse.
- **Metrics**: P95 TTFT, P25/median output speed, steps/min, throughput (aligned with AA).

## Usage

```bash
# Arm A: program-aware router
python aa_loadgen.py --url http://localhost:8100/v1 --model <MODEL> \
    --concurrency 64 --duration 600 --arm A --out aa_armA.json

# Arm B: KV-routing-only baseline (same load, swap the router)
python aa_loadgen.py --url http://localhost:8100/v1 --model <MODEL> \
    --concurrency 64 --duration 600 --arm B --out aa_armB.json
```

Requires `aiohttp`. Target any OpenAI-compatible `/v1/chat/completions` endpoint.

## Why synthetic load

Real agent harnesses (e.g. mini-SWE-agent on SWE-bench) couple "can the model solve
the task" with "how well does the router schedule", produce non-reproducible
trajectories, and require per-instance Docker sandboxes. A synthetic AA-style
generator gives a **controllable, reproducible** workload whose ISL can be tuned to
saturate the KV cache and trigger scheduler pause/resume — isolating the routing
layer as the only variable in an A/B comparison.
