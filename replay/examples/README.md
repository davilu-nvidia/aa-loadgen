# Example trajectory

One real recorded session: DSV4-Flash solving SWE-bench Lite
`astropy__astropy-12907` via mini-SWE-agent (25 turns).

Two files — source vs replay-ready:

| file | granularity | purpose |
|---|---|---|
| `astropy-12907.dsv4flash.raw.jsonl` | one line per **turn** (25) | **source of truth** from the recording hook. Keeps `output` text, raw timestamps (`request_ts`/`first_token_ts`/`last_token_ts`), and engine `prompt_tokens`/`completion_tokens`. Re-convert from this if the converter changes. |
| `astropy-12907.dsv4flash.dag.jsonl` | one line per **session** (1) | **replay input** (aiperf dag_jsonl). Derived by `convert_to_dag.py`. Drops `output` text; timestamps collapsed into per-turn `delay`. |

Conversion is lossy (dag drops output text + raw timestamps), so the raw file is the
archival source; the dag file is the generated artifact.

Load characteristics:
- ISL grows 1607 -> 20616 tokens across 25 turns (context accumulation)
- OSL 22..2048 tokens/turn (short pytest confirmations vs long code writes)
- inter-turn tool delay: median ~0.5s (lightweight cat/sed/grep/pytest)
- `recorded_isl` / `max_tokens` are authoritative engine usage (prompt/completion_tokens)

Replay:
```bash
python ../aa_replay.py --url http://localhost:8000/v1 --model dsv4flash \
    --replay astropy-12907.dsv4flash.dag.jsonl --concurrency 1 --arm demo
```

Re-convert from raw:
```bash
python ../convert_to_dag.py --in astropy-12907.dsv4flash.raw.jsonl \
    --out astropy-12907.dsv4flash.dag.jsonl --model-name dsv4flash
```
