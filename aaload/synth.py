"""aaload.synth — AA-AgentPerf-style *synthetic* multi-turn agentic load.

Each virtual agent = one program (session): N sequential turns with a growing
conversation context, simulated inter-turn tool latency, per-session nonce in
the system prompt (defeats cross-arm KV/prefix reuse for a fair A/B), and an
x-dynamo-session-id header for program-aware routing.
"""
import asyncio
import math
import random
import time

import aiohttp

from . import core

# ---------- AA-AgentPerf distribution parameters ----------
ISL_MEAN = 27000                     # AA: mean ~27K (per-request)
ISL_MIN, ISL_MAX = 5000, 131000
TOOL_DELAY_MED = 1.0                 # AA: median 1s
TOOL_DELAY_MIN, TOOL_DELAY_MAX = 0.1, 5.0
TURNS_MIN, TURNS_MAX = 6, 40         # turns per session (AA mentions up to ~200)

# Code-like filler used to reach a target ISL (approx 4 chars/token).
FILLER_LINE = ("def process_item(self, item, ctx, idx):  # module-level helper for the data pipeline stage\n"
               "    result = transform(item) if item is not None else default_value(ctx, idx)\n"
               "    return validate(result, schema=ctx.schema) or fallback(item, idx)\n")


def sample_isl(rng):
    """Target *per-request mean* ISL: lognormal, mean 27K, truncated [5K, 131K]."""
    mu, sigma = math.log(ISL_MEAN) - 0.5 * 0.7 ** 2, 0.7
    for _ in range(20):
        v = int(rng.lognormvariate(mu, sigma))
        if ISL_MIN <= v <= ISL_MAX:
            return v
    return ISL_MEAN


def sample_tool_delay(rng):
    v = rng.lognormvariate(math.log(TOOL_DELAY_MED), 0.8)
    return max(TOOL_DELAY_MIN, min(TOOL_DELAY_MAX, v))


def sample_osl(rng):
    # Mixture: 60% short (tool call, 20-120 tok) + 40% long (reasoning, 200-800 tok).
    if rng.random() < 0.6:
        return rng.randint(20, 120)
    return rng.randint(200, 800)


def make_filler(n_tokens):
    n_chars = n_tokens * 4
    reps = max(1, n_chars // len(FILLER_LINE))
    return (FILLER_LINE * reps)[:n_chars]


async def run_session(sess_idx, args, rng, metrics, stop_ts, http):
    """One virtual agent: sequential multi-turn, growing context, tool delays."""
    nonce = f"{args.arm}-s{sess_idx}-{args.run_id}"
    session_id = f"aa-{nonce}"
    n_turns = rng.randint(TURNS_MIN, TURNS_MAX)
    # Context ramps roughly linearly from peak/n to peak across the session, so
    # the per-request mean is ~peak*(n+1)/(2n). Choose peak so the *mean* matches
    # the sampled AA target (previously peak == sample, making the realized mean
    # about half the documented 27K).
    mean_isl = sample_isl(rng)
    peak_isl = min(ISL_MAX, int(mean_isl * 2 * n_turns / (n_turns + 1)))
    messages = [
        {"role": "system",
         "content": f"[session-nonce {nonce}] You are a coding agent solving a task."},
        {"role": "user",
         "content": "Task: fix the failing test.\n" + make_filler(peak_isl // n_turns)},
    ]
    headers = {"Content-Type": "application/json",
               "x-dynamo-session-id": session_id}
    cur_isl = peak_isl // n_turns
    t_sess = time.time()
    for turn in range(n_turns):
        if time.time() > stop_ts:
            break
        if turn > 0:
            # Simulated tool latency between turns (none before the first turn,
            # none after the last — a trailing sleep only wastes the slot).
            await asyncio.sleep(sample_tool_delay(rng))
        osl = sample_osl(rng)
        body = {"model": args.model, "messages": messages,
                "max_tokens": osl, "temperature": 0.0, "stream": True}
        if args.ignore_eos:
            # Force exactly `osl` output tokens so the realized OSL follows the
            # sampled distribution instead of wherever the model stops.
            body["ignore_eos"] = True
            body["min_tokens"] = osl
        r = await core.stream_chat(http, args.url, body, headers, args.stream_usage)
        if not r["ok"]:
            metrics.record_err()
            break  # synthetic context depends on the reply -> abort the session
        metrics.record_ok(r["ttft"], r["out_tok"], r["dt"])
        # Grow context: assistant reply + tool result padded toward next ISL target.
        messages.append({"role": "assistant", "content": r["content"] or "(step)"})
        grow = (peak_isl - cur_isl) // max(1, n_turns - turn)
        messages.append({"role": "user",
                         "content": "[tool output]\n" + make_filler(max(200, grow))})
        cur_isl += grow
    metrics.record_session(time.time() - t_sess)


async def run(args):
    stop_ts = time.time() + args.duration
    metrics = core.Metrics(warmup_until=time.time() + args.warmup)
    conn = aiohttp.TCPConnector(limit=0)
    out = args.out or "/tmp/aa_result.json"
    async with aiohttp.ClientSession(connector=conn) as http:
        sess_counter = [0]

        async def worker(slot):
            r = random.Random(args.seed + slot)
            while time.time() < stop_ts:
                sess_counter[0] += 1
                await run_session(sess_counter[0], args, r, metrics, stop_ts, http)

        print(f"[synth] arm={args.arm} concurrency={args.concurrency} "
              f"duration={args.duration}s warmup={args.warmup}s -> {args.url} model={args.model}")
        t_start = time.time()
        await asyncio.gather(*[worker(i) for i in range(args.concurrency)])
        elapsed = time.time() - t_start

    res = core.build_report(metrics, elapsed, {
        "mode": "synth", "arm": args.arm, "concurrency": args.concurrency,
    })
    core.save_report(res, out)


def add_parser(sub):
    p = sub.add_parser("synth", help="synthesize AA-AgentPerf-style multi-turn load")
    core.add_common_args(p)
    p.add_argument("--model", default="DeepSeek-V4-Flash-FP8")
    p.add_argument("--duration", type=int, default=600,
                   help="measurement window (s)")
    p.add_argument("--run-id", default="1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ignore-eos", action="store_true",
                   help="force exact OSL via ignore_eos+min_tokens (vLLM/SGLang extension)")
    p.set_defaults(func=run)
    return p
