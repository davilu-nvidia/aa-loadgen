#!/usr/bin/env python3
"""
AA-loadgen REPLAY mode: replay recorded agentic trajectories (dag_jsonl) against
an OpenAI-compatible endpoint. Companion to aa_loadgen.py (which synthesizes load);
this one replays REAL recorded sessions for faithful A/B of the Agentic Infra.

- Reads dag_jsonl: one session per line, {session_id, turns:[{messages, max_tokens, model, delay}]}
- Closed-loop: keeps --concurrency sessions in flight; each slot pulls the next
  session from the pool until the pool is exhausted (or --duration elapses).
- Per session: replays each turn's recorded `messages` verbatim, max_tokens = recorded OSL,
  sleeps `delay` ms between turns (the real tool wall-clock). Injects x-dynamo-session-id
  = session_id so the program-aware router groups turns.
- Pure request-replay: model output is consumed for metrics only (TTFT / output speed),
  not fed back (the next turn's messages are already recorded).
- Metrics aligned with AA: P95 TTFT, P25/median output speed, steps/min, throughput.

Usage:
  python aa_replay.py --url http://localhost:8000/v1 --model dsv4flash \
      --replay dsv4flash.dag.jsonl --concurrency 64 --arm A --out armA.json
"""
import asyncio, aiohttp, time, json, argparse, itertools

def load_sessions(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

class Metrics:
    def __init__(self):
        self.ttft = []; self.out_speed = []
        self.req_ok = 0; self.req_err = 0
        self.total_out_tokens = 0; self.steps = 0; self.sessions_done = 0

async def replay_session(sess, args, metrics, stop_ts):
    session_id = sess["session_id"]
    headers = {"Content-Type": "application/json", "x-dynamo-session-id": session_id}
    for ti, turn in enumerate(sess["turns"]):
        if stop_ts and time.time() > stop_ts:
            break
        # inter-turn tool delay (recorded real wall-clock), skip before first turn
        if ti > 0:
            await asyncio.sleep(turn.get("delay", 0.0) / 1000.0)
        _osl = int(turn.get("max_tokens", 256))
        body = {
            "model": args.model or turn.get("model", "model"),
            "messages": turn["messages"],
            "max_tokens": _osl,
            "temperature": 0.0,
            "stream": True,
        }
        if args.ignore_eos:
            # Force the model to emit exactly _osl tokens so replayed OSL matches
            # the recorded value (SGLang/vLLM honor ignore_eos + min_tokens).
            body["ignore_eos"] = True
            body["min_tokens"] = _osl
        t0 = time.time(); ttft = None; out_tok = 0
        try:
            async with args.session.post(f"{args.url}/chat/completions", json=body,
                                         headers=headers,
                                         timeout=aiohttp.ClientTimeout(total=600)) as resp:
                if resp.status != 200:
                    metrics.req_err += 1; await resp.read()
                    continue  # don't kill whole session on one bad turn
                async for raw in resp.content:
                    line = raw.decode("utf-8", "ignore").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        j = json.loads(data)
                        ch = j.get("choices", [{}])[0]
                        delta = ch.get("delta", {}).get("content")
                        usage = j.get("usage")
                        if delta:
                            if ttft is None:
                                ttft = time.time() - t0
                            out_tok += 1
                        if usage and isinstance(usage.get("completion_tokens"), int):
                            out_tok = usage["completion_tokens"]  # authoritative if present
                    except Exception:
                        continue
        except Exception:
            metrics.req_err += 1
            continue
        dt = time.time() - t0
        if ttft is not None and out_tok > 0:
            metrics.ttft.append(ttft)
            metrics.out_speed.append(out_tok / max(1e-3, dt - ttft))
            metrics.total_out_tokens += out_tok
            metrics.req_ok += 1; metrics.steps += 1
        else:
            metrics.req_err += 1
    metrics.sessions_done += 1

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default=None, help="override model; default uses per-turn recorded model")
    ap.add_argument("--replay", required=True, help="dag_jsonl trajectory file")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--duration", type=int, default=0, help="0 = replay whole pool once; >0 = cap seconds")
    ap.add_argument("--loop", action="store_true", help="loop the pool until duration elapses")
    ap.add_argument("--arm", default="A")
    ap.add_argument("--ignore-eos", action="store_true", default=True, help="force exact OSL via ignore_eos+min_tokens")
    ap.add_argument("--no-ignore-eos", dest="ignore_eos", action="store_false")
    ap.add_argument("--out", default="/tmp/aa_replay_result.json")
    args = ap.parse_args()

    sessions = load_sessions(args.replay)
    print(f"[replay] loaded {len(sessions)} sessions from {args.replay}")
    stop_ts = time.time() + args.duration if args.duration > 0 else None
    metrics = Metrics()
    conn = aiohttp.TCPConnector(limit=0)

    # work queue
    pool = itertools.cycle(sessions) if args.loop else iter(sessions)
    lock = asyncio.Lock()
    async def next_session():
        async with lock:
            try:
                return next(pool)
            except StopIteration:
                return None

    async with aiohttp.ClientSession(connector=conn) as session:
        args.session = session
        async def worker():
            while True:
                if stop_ts and time.time() > stop_ts:
                    return
                s = await next_session()
                if s is None:
                    return
                await replay_session(s, args, metrics, stop_ts)
        print(f"[replay] arm={args.arm} concurrency={args.concurrency} -> {args.url}")
        t_start = time.time()
        await asyncio.gather(*[worker() for _ in range(args.concurrency)])
        elapsed = time.time() - t_start

    def pct(a, p):
        if not a: return 0
        s = sorted(a); return s[min(len(s)-1, int(len(s)*p))]
    res = {
        "arm": args.arm, "mode": "replay", "concurrency": args.concurrency,
        "elapsed_s": round(elapsed, 1), "sessions_done": metrics.sessions_done,
        "requests_ok": metrics.req_ok, "requests_err": metrics.req_err,
        "total_steps": metrics.steps,
        "steps_per_min": round(metrics.steps / (elapsed/60), 1) if elapsed else 0,
        "total_out_tokens": metrics.total_out_tokens,
        "throughput_tok_s": round(metrics.total_out_tokens / elapsed, 1) if elapsed else 0,
        "ttft_p50_s": round(pct(metrics.ttft, 0.50), 3),
        "ttft_p95_s": round(pct(metrics.ttft, 0.95), 3),
        "out_speed_p25_tok_s": round(pct(metrics.out_speed, 0.25), 1),
        "out_speed_median_tok_s": round(pct(metrics.out_speed, 0.50), 1),
    }
    print(json.dumps(res, indent=2, ensure_ascii=False))
    json.dump(res, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"[saved] {args.out}")

if __name__ == "__main__":
    asyncio.run(main())
