"""aaload.core — shared plumbing for the synth and replay load generators.

Streaming OpenAI-compatible client, metrics accumulation, percentile helpers
and the report writer. Both modes (aaload.synth / aaload.replay) build on
these primitives so they measure identically.
"""
import json
import time

import aiohttp

REQUEST_TIMEOUT_S = 300


def pct(a, p):
    """Nearest-rank percentile; 0 when empty."""
    if not a:
        return 0
    s = sorted(a)
    return s[min(len(s) - 1, int(len(s) * p))]


class Metrics:
    """Per-request sample accumulator.

    Samples recorded before `warmup_until` (unix ts) are discarded so the
    report reflects steady state rather than the cold-start / cache-warming
    transient.
    """

    def __init__(self, warmup_until=0.0):
        self.warmup_until = warmup_until
        self.ttft = []          # s
        self.out_speed = []     # tok/s (decode phase)
        self.tpot = []          # ms/token
        self.e2e = []           # s, per session
        self.req_ok = 0
        self.req_err = 0
        self.total_out_tokens = 0
        self.steps = 0          # completed agent steps (= successful requests)
        self.sessions_done = 0

    def in_warmup(self):
        return time.time() < self.warmup_until

    def record_ok(self, ttft, out_tok, dt):
        if self.in_warmup():
            return
        decode_t = max(1e-3, dt - ttft)
        self.ttft.append(ttft)
        self.out_speed.append(out_tok / decode_t)
        self.tpot.append(decode_t / max(1, out_tok) * 1000.0)
        self.total_out_tokens += out_tok
        self.req_ok += 1
        self.steps += 1

    def record_err(self):
        if self.in_warmup():
            return
        self.req_err += 1

    def record_session(self, e2e_s):
        self.sessions_done += 1
        if not self.in_warmup():
            self.e2e.append(e2e_s)


async def stream_chat(http, url, body, headers, stream_usage=True):
    """POST a streaming /chat/completions request and parse the SSE stream.

    Returns a dict: {ok, ttft, out_tok, content, dt}.

    Token counting: requests a final usage chunk via stream_options
    (include_usage) and prefers the server-reported completion_tokens.
    Falls back to counting content deltas, which under-counts on servers
    that batch multiple tokens per SSE chunk — hence the usage preference.
    """
    body = dict(body)
    if stream_usage:
        so = dict(body.get("stream_options") or {})
        so["include_usage"] = True
        body["stream_options"] = so
    t0 = time.time()
    ttft = None
    chunk_tok = 0
    usage_tok = None
    parts = []
    try:
        async with http.post(f"{url}/chat/completions", json=body, headers=headers,
                             timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S)) as resp:
            if resp.status != 200:
                await resp.read()
                return {"ok": False, "ttft": None, "out_tok": 0, "content": "",
                        "dt": time.time() - t0}
            async for raw in resp.content:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    j = json.loads(data)
                except Exception:
                    continue
                choices = j.get("choices") or []
                delta = choices[0].get("delta", {}).get("content") if choices else None
                if delta:
                    if ttft is None:
                        ttft = time.time() - t0
                    chunk_tok += 1
                    parts.append(delta)
                usage = j.get("usage")
                if usage and isinstance(usage.get("completion_tokens"), int):
                    usage_tok = usage["completion_tokens"]  # authoritative
    except Exception:
        return {"ok": False, "ttft": None, "out_tok": 0, "content": "",
                "dt": time.time() - t0}
    dt = time.time() - t0
    out_tok = usage_tok if usage_tok else chunk_tok
    return {"ok": ttft is not None and out_tok > 0, "ttft": ttft,
            "out_tok": out_tok, "content": "".join(parts), "dt": dt}


def add_common_args(p):
    p.add_argument("--url", default="http://localhost:8000/v1")
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--arm", default="A", help="experiment arm label (A/B)")
    p.add_argument("--warmup", type=int, default=0,
                   help="discard samples from the first N seconds (steady-state report)")
    p.add_argument("--no-stream-usage", dest="stream_usage", action="store_false",
                   help="don't request a usage chunk (for servers that reject stream_options)")
    p.add_argument("--out", default=None, help="result json path")


def build_report(metrics, elapsed, extra):
    res = dict(extra)
    res.update({
        "elapsed_s": round(elapsed, 1),
        "sessions_done": metrics.sessions_done,
        "requests_ok": metrics.req_ok,
        "requests_err": metrics.req_err,
        "total_steps": metrics.steps,
        "steps_per_min": round(metrics.steps / (elapsed / 60), 1) if elapsed else 0,
        "total_out_tokens": metrics.total_out_tokens,
        "throughput_tok_s": round(metrics.total_out_tokens / elapsed, 1) if elapsed else 0,
        "ttft_p50_s": round(pct(metrics.ttft, 0.50), 3),
        "ttft_p95_s": round(pct(metrics.ttft, 0.95), 3),
        "tpot_p50_ms": round(pct(metrics.tpot, 0.50), 1),
        "tpot_p95_ms": round(pct(metrics.tpot, 0.95), 1),
        "e2e_p50_s": round(pct(metrics.e2e, 0.50), 1),
        "e2e_p95_s": round(pct(metrics.e2e, 0.95), 1),
        "out_speed_p25_tok_s": round(pct(metrics.out_speed, 0.25), 1),  # AA uses P25
        "out_speed_median_tok_s": round(pct(metrics.out_speed, 0.50), 1),
    })
    return res


def save_report(res, path):
    print(json.dumps(res, indent=2, ensure_ascii=False))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"[saved] {path}")
