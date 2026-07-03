#!/usr/bin/env python3
"""
AA-AgentPerf-style load generator (for Dynamo thunderagent_router A/B)
=====================================================================
Synthesizes multi-turn agentic load following the Artificial Analysis
AA-AgentPerf methodology:
  - Each virtual agent = one program (session): issues N sequential turns,
    carrying a growing conversation context.
  - ISL 5K-131K, mean ~27K (lognormal), grows across turns.
  - OSL variable (short tool-calls vs long reasoning).
  - Inter-turn tool delay injected (median 1s, range 0.1-5s); tools are NOT
    actually executed (delay is simulated, matching AA's replay model).
  - Injects x-dynamo-session-id so thunderagent_router schedules per program.
  - Per-session unique nonce prefix (prevents cross-arm KV reuse for a fair A/B).
  - Metrics: P95 TTFT, P25/median output speed, steps/min, throughput.

Usage:
  python aa_loadgen.py --url http://localhost:8100/v1 --model DeepSeek-V4-Flash-FP8 \
      --concurrency 64 --duration 600 --arm A --out /tmp/aa_armA.json
"""
import asyncio, aiohttp, time, random, json, argparse, math

# ---------- AA-AgentPerf distribution parameters ----------
ISL_MEAN = 27000                     # AA: mean ~27K
ISL_MIN, ISL_MAX = 5000, 131000
TOOL_DELAY_MED = 1.0                 # AA: median 1s
TOOL_DELAY_MIN, TOOL_DELAY_MAX = 0.1, 5.0
TURNS_MIN, TURNS_MAX = 6, 40         # turns per session (AA mentions up to ~200)

# Code-like filler text used to reach a target ISL (approx 4 chars/token).
FILLER_LINE = ("def process_item(self, item, ctx, idx):  # module-level helper for the data pipeline stage\n"
               "    result = transform(item) if item is not None else default_value(ctx, idx)\n"
               "    return validate(result, schema=ctx.schema) or fallback(item, idx)\n")


def sample_isl(rng):
    # Lognormal fit for mean 27K, truncated to [5K, 131K].
    mu, sigma = math.log(ISL_MEAN) - 0.5 * 0.7 ** 2, 0.7
    for _ in range(20):
        v = int(rng.lognormvariate(mu, sigma))
        if ISL_MIN <= v <= ISL_MAX:
            return v
    return ISL_MEAN


def sample_tool_delay(rng):
    # Lognormal, median ~1s, truncated to [0.1, 5].
    v = rng.lognormvariate(math.log(TOOL_DELAY_MED), 0.8)
    return max(TOOL_DELAY_MIN, min(TOOL_DELAY_MAX, v))


def sample_osl(rng):
    # Mixture: 60% short (tool call, 20-120 tok) + 40% long (reasoning, 200-800 tok).
    if rng.random() < 0.6:
        return rng.randint(20, 120)
    return rng.randint(200, 800)


def make_filler(n_tokens):
    # ~4 chars/token; repeat code-like lines to reach the target size.
    n_chars = n_tokens * 4
    reps = max(1, n_chars // len(FILLER_LINE))
    return (FILLER_LINE * reps)[:n_chars]


class Metrics:
    def __init__(self):
        self.ttft = []              # per-request time-to-first-token (s)
        self.out_speed = []         # per-request output speed (tok/s)
        self.req_ok = 0
        self.req_err = 0
        self.total_out_tokens = 0
        self.steps = 0              # completed agent steps (= successful requests)
        self.sessions_done = 0


async def run_session(sess_idx, args, rng, metrics, stop_ts):
    """One virtual agent: sequential multi-turn, growing context, inter-turn tool delay."""
    nonce = f"{args.arm}-s{sess_idx}-{args.run_id}"
    session_id = f"aa-{nonce}"
    peak_isl = sample_isl(rng)
    n_turns = rng.randint(TURNS_MIN, TURNS_MAX)
    # Initial messages: system (with nonce to defeat cross-arm caching) + user.
    messages = [
        {"role": "system", "content": f"[session-nonce {nonce}] You are a coding agent solving a task."},
        {"role": "user", "content": "Task: fix the failing test.\n" + make_filler(peak_isl // n_turns)},
    ]
    headers = {"Content-Type": "application/json",
               "x-dynamo-session-id": session_id}
    cur_isl = peak_isl // n_turns
    for turn in range(n_turns):
        if time.time() > stop_ts:
            break
        osl = sample_osl(rng)
        body = {"model": args.model, "messages": messages,
                "max_tokens": osl, "temperature": 0.0, "stream": True}
        t0 = time.time(); ttft = None; out_tok = 0; content_parts = []
        try:
            async with args.session.post(f"{args.url}/chat/completions",
                                         json=body, headers=headers,
                                         timeout=aiohttp.ClientTimeout(total=300)) as resp:
                if resp.status != 200:
                    metrics.req_err += 1
                    await resp.read()
                    break
                async for raw in resp.content:
                    line = raw.decode("utf-8", "ignore").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        j = json.loads(data)
                        delta = j["choices"][0].get("delta", {}).get("content")
                        if delta:
                            if ttft is None:
                                ttft = time.time() - t0
                            out_tok += 1
                            content_parts.append(delta)
                    except Exception:
                        continue
        except Exception:
            metrics.req_err += 1
            break
        dt = time.time() - t0
        if ttft is not None and out_tok > 0:
            metrics.ttft.append(ttft)
            decode_t = max(1e-3, dt - ttft)
            metrics.out_speed.append(out_tok / decode_t)
            metrics.total_out_tokens += out_tok
            metrics.req_ok += 1
            metrics.steps += 1
        else:
            metrics.req_err += 1
            break
        # Grow context: assistant reply + tool result (padded toward next-turn ISL target).
        messages.append({"role": "assistant", "content": "".join(content_parts) or "(step)"})
        grow = (peak_isl - cur_isl) // max(1, n_turns - turn)
        messages.append({"role": "user", "content": "[tool output]\n" + make_filler(max(200, grow))})
        cur_isl += grow
        # Inter-turn tool delay (tool is not actually executed).
        await asyncio.sleep(sample_tool_delay(rng))
    metrics.sessions_done += 1


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8100/v1")
    ap.add_argument("--model", default="DeepSeek-V4-Flash-FP8")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--duration", type=int, default=600, help="steady-state measurement window (s)")
    ap.add_argument("--arm", default="A", help="A=program-aware, B=kv-only (used for nonce/label)")
    ap.add_argument("--run-id", default="1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="/tmp/aa_result.json")
    args = ap.parse_args()

    stop_ts = time.time() + args.duration
    metrics = Metrics()
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=conn) as session:
        args.session = session
        # Keep `concurrency` sessions in flight: each slot starts a new session as soon
        # as the previous one finishes.
        sess_counter = [0]

        async def worker(slot):
            r = random.Random(args.seed + slot)
            while time.time() < stop_ts:
                sess_counter[0] += 1
                await run_session(sess_counter[0], args, r, metrics, stop_ts)

        print(f"[AA-loadgen] arm={args.arm} concurrency={args.concurrency} "
              f"duration={args.duration}s -> {args.url} model={args.model}")
        t_start = time.time()
        await asyncio.gather(*[worker(i) for i in range(args.concurrency)])
        elapsed = time.time() - t_start

    # Aggregate.
    def pct(a, p):
        if not a:
            return 0
        s = sorted(a)
        return s[min(len(s) - 1, int(len(s) * p))]

    res = {
        "arm": args.arm, "concurrency": args.concurrency, "elapsed_s": round(elapsed, 1),
        "sessions_done": metrics.sessions_done,
        "requests_ok": metrics.req_ok, "requests_err": metrics.req_err,
        "total_steps": metrics.steps,
        "steps_per_min": round(metrics.steps / (elapsed / 60), 1),
        "total_out_tokens": metrics.total_out_tokens,
        "throughput_tok_s": round(metrics.total_out_tokens / elapsed, 1),
        "ttft_p50_s": round(pct(metrics.ttft, 0.50), 3),
        "ttft_p95_s": round(pct(metrics.ttft, 0.95), 3),
        "out_speed_p25_tok_s": round(pct(metrics.out_speed, 0.25), 1),   # AA uses P25
        "out_speed_median_tok_s": round(pct(metrics.out_speed, 0.50), 1),
    }
    print(json.dumps(res, indent=2, ensure_ascii=False))
    json.dump(res, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
