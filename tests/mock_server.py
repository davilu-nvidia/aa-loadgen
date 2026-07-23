#!/usr/bin/env python3
"""Mock OpenAI-compatible SSE server for smoke-testing aa-loadgen.

Streams 2 "tokens" per SSE chunk (so chunk-counting under-counts by 2x) and
emits a final usage chunk — verifies the usage-based counting path.

  python tests/mock_server.py [port]
"""
import asyncio
import json
import sys

from aiohttp import web


async def chat(req):
    body = await req.json()
    n = min(int(body.get("max_tokens", 16)), 64)
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(req)
    sent = 0
    while sent < n:
        k = min(2, n - sent)
        chunk = {"choices": [{"delta": {"content": "tok " * k}}]}
        await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
        sent += k
        await asyncio.sleep(0.003)
    usage = {"choices": [],
             "usage": {"prompt_tokens": 100, "completion_tokens": n}}
    await resp.write(f"data: {json.dumps(usage)}\n\n".encode())
    await resp.write(b"data: [DONE]\n\n")
    await resp.write_eof()
    return resp


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9009
    app = web.Application(client_max_size=1024 ** 3)
    app.router.add_post("/v1/chat/completions", chat)
    web.run_app(app, port=port)


if __name__ == "__main__":
    main()
