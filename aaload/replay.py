"""aaload.replay — replay recorded agentic trajectories (dag_jsonl).

Reads dag_jsonl (one session per line: {session_id, turns:[{messages,
max_tokens, model, delay}]}) and replays each turn's recorded messages
verbatim, with the recorded inter-turn tool wall-clock. Pure request-replay:
model output is consumed for metrics only, never fed back.
"""
import asyncio
import itertools
import json
import time

import aiohttp

from . import core


def load_sessions(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def with_nonce(messages, nonce_text):
    """Copy `messages` with the replica nonce prefixed to the first message.

    Engine prefix caches (vLLM/SGLang) key on the *token sequence*; a nonce
    in x-dynamo-session-id alone does not break cross-replica KV sharing when
    the pool is looped. The nonce must change the content. Original recorded
    dicts are never mutated (they're shared across loop replicas).
    """
    msgs = list(messages)
    if msgs and isinstance(msgs[0].get("content"), str):
        first = dict(msgs[0])
        first["content"] = nonce_text + first["content"]
        msgs[0] = first
    else:
        msgs.insert(0, {"role": "system", "content": nonce_text.strip()})
    return msgs


def write_progress(metrics, path):
    try:
        p50 = lambda a: core.pct(a, 0.50)
        with open(path, "w") as f:
            f.write(f"{metrics.sessions_done} done, ok={metrics.req_ok} "
                    f"err={metrics.req_err} steps={metrics.steps} "
                    f"ttft_p50={p50(metrics.ttft):.3f} "
                    f"tpot_p50={p50(metrics.tpot):.1f} "
                    f"e2e_p50={p50(metrics.e2e):.1f}\n")
    except Exception:
        pass


async def replay_session(sess, args, metrics, stop_ts, http, run_idx):
    base_sid = sess["session_id"]
    # Unique id per loop replica so the router treats copies as distinct programs.
    session_id = f"{base_sid}#r{run_idx}" if args.nonce else base_sid
    nonce_text = f"[replica-nonce {session_id}] "
    headers = {"Content-Type": "application/json"}
    if args.agent_context:
        headers["x-dynamo-session-id"] = session_id
    t_sess = time.time()
    n_turns = len(sess["turns"])
    for ti, turn in enumerate(sess["turns"]):
        if stop_ts and time.time() > stop_ts:
            break
        # Recorded inter-turn tool wall-clock; skip before the first turn.
        if ti > 0:
            await asyncio.sleep(turn.get("delay", 0.0) / 1000.0)
        req_headers = headers
        if args.agent_context and ti == n_turns - 1:
            req_headers = dict(headers)
            req_headers["x-dynamo-session-final"] = "true"
        messages = turn["messages"]
        if args.nonce:
            messages = with_nonce(messages, nonce_text)
        osl = int(turn.get("max_tokens", 256))
        body = {"model": args.model or turn.get("model", "model"),
                "messages": messages, "max_tokens": osl,
                "temperature": 0.0, "stream": True}
        if args.ignore_eos:
            # Reproduce the recorded OSL exactly (vLLM/SGLang extension).
            body["ignore_eos"] = True
            body["min_tokens"] = osl
        r = await core.stream_chat(http, args.url, body, req_headers, args.stream_usage)
        if r["ok"]:
            metrics.record_ok(r["ttft"], r["out_tok"], r["dt"])
        else:
            metrics.record_err()
            continue  # recorded context stays valid; don't kill the session
    metrics.record_session(time.time() - t_sess)
    write_progress(metrics, args.progress_file)


async def run(args):
    sessions = load_sessions(args.replay)
    print(f"[replay] loaded {len(sessions)} sessions from {args.replay}")
    out = args.out or "/tmp/aa_replay_result.json"
    if not args.progress_file:
        args.progress_file = out + ".progress"  # per-run path: no cross-arm clobber
    stop_ts = time.time() + args.duration if args.duration > 0 else None
    metrics = core.Metrics(warmup_until=time.time() + args.warmup)
    conn = aiohttp.TCPConnector(limit=0)

    pool = itertools.cycle(sessions) if args.loop else iter(sessions)
    run_counter = itertools.count()
    lock = asyncio.Lock()

    async def next_session():
        async with lock:
            try:
                return next(pool)
            except StopIteration:
                return None

    async with aiohttp.ClientSession(connector=conn) as http:
        async def worker():
            while True:
                if stop_ts and time.time() > stop_ts:
                    return
                s = await next_session()
                if s is None:
                    return
                await replay_session(s, args, metrics, stop_ts, http,
                                     run_idx=next(run_counter))

        print(f"[replay] arm={args.arm} concurrency={args.concurrency} -> {args.url}")
        t_start = time.time()
        tasks = [asyncio.create_task(worker()) for _ in range(args.concurrency)]
        try:
            hard = (args.duration + 45) if args.duration > 0 else None
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True),
                                   timeout=hard)
        except asyncio.TimeoutError:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            print("[replay] hard deadline hit, forced finish")
        elapsed = time.time() - t_start

    res = core.build_report(metrics, elapsed, {
        "mode": "replay", "arm": args.arm, "concurrency": args.concurrency,
    })
    core.save_report(res, out)


def add_parser(sub):
    p = sub.add_parser("replay", help="replay recorded dag_jsonl trajectories")
    core.add_common_args(p)
    p.add_argument("--model", default=None,
                   help="override model; default uses per-turn recorded model")
    p.add_argument("--replay", required=True, help="dag_jsonl trajectory file")
    p.add_argument("--duration", type=int, default=0,
                   help="0 = replay whole pool once; >0 = cap seconds")
    p.add_argument("--loop", action="store_true",
                   help="loop the pool until duration elapses")
    p.add_argument("--nonce", action="store_true",
                   help="per-replica nonce in message content + session id "
                        "(breaks cross-copy prefix sharing for real KV pressure)")
    p.add_argument("--agent-context", action="store_true",
                   help="send x-dynamo-session-id to trigger program scheduling")
    p.add_argument("--ignore-eos", action="store_true", default=True,
                   help="force exact OSL via ignore_eos+min_tokens (default on)")
    p.add_argument("--no-ignore-eos", dest="ignore_eos", action="store_false")
    p.add_argument("--progress-file", default=None,
                   help="live progress path (default: <out>.progress)")
    p.set_defaults(func=run)
    return p
