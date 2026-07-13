# Example trajectory

One real recorded session: DSV4-Flash solving SWE-bench Lite
`astropy__astropy-12907` via mini-SWE-agent (25 turns).

Two files — source vs replay-ready:

| file | granularity | purpose |
|---|---|---|
| `astropy-12907.dsv4flash.raw.jsonl` | one line per **turn** (25) | **source of truth** from the recording hook. Keeps `output` text, raw timestamps (`request_ts`/`first_token_ts`/`last_token_ts`), and engine `prompt_tokens`/`completion_tokens`. Re-convert from this if the converter changes. |
| `astropy-12907.dsv4flash.dag.jsonl` | one line per **session** (1) | **replay input** (aiperf dag_jsonl). Derived by `convert_to_dag.py`. Drops `output` text; timestamps collapsed into per-turn `delay`. |

The dag file validates against aiperf's strict `DagConversation` schema (custom `recorded_isl` lives under `extra`), so it is aiperf-compatible: run it with
`aiperf profile --input-file *.dag.jsonl --custom-dataset-type dag_jsonl ...`.

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

## aiperf compatibility (verified end-to-end)

`astropy-12907.dsv4flash.aiperf.dag.jsonl` — the same session in **incremental**
form (each turn carries only NEW messages; `system` on root turn only), which is
what aiperf's strict pure-append DagConversation requires. ~15x smaller than the
full-context `.dag.jsonl` since history is not repeated.

Verified end-to-end with aiperf 0.9.0:
```bash
aiperf profile -m dsv4flash --tokenizer /path/to/DeepSeek-V4-Flash \
    --endpoint-type chat --streaming --url http://localhost:8000 \
    --input-file astropy-12907.dsv4flash.aiperf.dag.jsonl \
    --custom-dataset-type dag_jsonl --concurrency 1
```
Report in `aiperf_report/`. Gotchas resolved: (1) don't set HF_HUB_OFFLINE (breaks
local tokenizer path); (2) use --incremental for aiperf (system-on-root rule);
(3) aiperf merges `extra` into the wire body, so no custom keys (recorded_isl is
dropped in incremental mode).

- `.dag.jsonl` (full context per turn)      -> aa_replay.py
- `.aiperf.dag.jsonl` (incremental)          -> aiperf
