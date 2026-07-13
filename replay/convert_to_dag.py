#!/usr/bin/env python3
"""
Convert recorded mini-SWE-agent trajectory turns -> aiperf dag_jsonl replay format.

Input : trajectory_turns.jsonl (one line per turn, from vllm_model recording hook)
Output: <model>.dag.jsonl (one line per session/trajectory)

Each output session:
  {"session_id": <trajectory_id>,
   "turns": [
     {"messages": [...as-sent...], "max_tokens": <real OSL>, "model": <name>,
      "delay": <inter-turn tool wall-clock ms>},
     ...]}

- max_tokens = real output length (tokenizer count of recorded `output`), so replay
  reproduces the true decode length per turn.
- delay = gap between previous turn's last_token_ts and this turn's request_ts
  (the real tool-execution wall clock). First turn delay=0.
- messages are recorded verbatim (the FULL accumulated context as actually sent),
  so replay is pure request-replay with real prefixes.

Usage:
  python convert_to_dag.py --in trajectory_turns.jsonl --out dsv4flash.dag.jsonl \
      --tokenizer /raid/model_hub/DeepSeek-V4-Flash
"""
import argparse, json, sys
from collections import defaultdict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--tokenizer", default=None, help="HF tokenizer path for exact OSL; falls back to len/4 chars")
    ap.add_argument("--model-name", default=None, help="override model name written per turn")
    args = ap.parse_args()

    tok = None
    if args.tokenizer:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
            print(f"[convert] tokenizer loaded: {args.tokenizer}", file=sys.stderr)
        except Exception as e:
            print(f"[convert] tokenizer load failed ({e}); using char/4 heuristic", file=sys.stderr)

    def osl(text):
        if not text:
            return 1
        if tok:
            try:
                return max(1, len(tok.encode(text, add_special_tokens=False)))
            except Exception:
                pass
        return max(1, len(text) // 4)

    # group turns by trajectory_id
    sessions = defaultdict(list)
    n = 0
    with open(args.inp, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            sessions[r["trajectory_id"]].append(r)
            n += 1
    print(f"[convert] read {n} turns across {len(sessions)} sessions", file=sys.stderr)

    n_out = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for sid, turns in sessions.items():
            turns.sort(key=lambda x: x.get("turn", 0))
            dag_turns = []
            prev_last_ts = None
            for t in turns:
                req_ts = t.get("request_ts")
                delay_ms = 0.0
                if prev_last_ts is not None and req_ts is not None:
                    delay_ms = max(0.0, (req_ts - prev_last_ts) * 1000.0)
                # Prefer authoritative engine usage (completion_tokens) for OSL;
                # fall back to tokenizer/char estimate only if absent.
                rec_osl = t.get("completion_tokens")
                osl_val = int(rec_osl) if isinstance(rec_osl, int) and rec_osl > 0 else osl(t.get("output", ""))
                turn_obj = {
                    "messages": t["messages"],
                    "max_tokens": osl_val,
                    "model": args.model_name or t.get("model_name", "model"),
                    "delay": round(delay_ms, 1),
                }
                # aiperf DagTurn is strict (extra="forbid"); custom metadata must
                # live under `extra`. recorded_isl = authoritative engine ISL.
                rec_isl = t.get("prompt_tokens")
                if rec_isl is not None:
                    turn_obj["extra"] = {"recorded_isl": rec_isl}
                dag_turns.append(turn_obj)
                prev_last_ts = t.get("last_token_ts")
            if dag_turns:
                out.write(json.dumps({"session_id": sid, "turns": dag_turns}, ensure_ascii=False) + "\n")
                n_out += 1
    print(f"[convert] wrote {n_out} sessions -> {args.out}", file=sys.stderr)

if __name__ == "__main__":
    main()
