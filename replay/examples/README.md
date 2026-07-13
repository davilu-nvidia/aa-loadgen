# Example trajectory

`astropy-12907.dsv4flash.dag.jsonl` — one real recorded session: DSV4-Flash solving
SWE-bench Lite `astropy__astropy-12907` via mini-SWE-agent (25 turns).

Load characteristics:
- ISL grows 1607 -> 20616 tokens across 25 turns (context accumulation)
- OSL 22..2048 tokens/turn (short pytest confirmations vs long code writes)
- inter-turn tool delay: median ~0.5s (lightweight cat/sed/grep/pytest)
- `recorded_isl` / `max_tokens` are authoritative engine usage (prompt/completion_tokens)

Replay it:
```bash
python ../aa_replay.py --url http://localhost:8000/v1 --model dsv4flash \
    --replay astropy-12907.dsv4flash.dag.jsonl --concurrency 1 --arm demo
```
