#!/usr/bin/env python3
"""aa-loadgen — multi-turn agentic inference load generator.

Two modes against any OpenAI-compatible /v1/chat/completions endpoint:

  synth   AA-AgentPerf-style synthetic load (controllable, reproducible;
          ISL/OSL/tool-delay distributions per the AA methodology)
  replay  faithful replay of recorded real trajectories (dag_jsonl from
          replay/convert_to_dag.py)

Examples:
  python aa_loadgen.py synth  --url http://localhost:8000/v1 --model M \\
      --concurrency 64 --duration 600 --arm A --out aa_armA.json
  python aa_loadgen.py replay --url http://localhost:8000/v1 --model M \\
      --replay traj.dag.jsonl --concurrency 64 --arm A --agent-context \\
      --out armA.json

Requires aiohttp.
"""
import argparse
import asyncio

from aaload import replay, synth


def main():
    ap = argparse.ArgumentParser(prog="aa_loadgen", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(required=True, metavar="{synth,replay}")
    synth.add_parser(sub)
    replay.add_parser(sub)
    args = ap.parse_args()
    asyncio.run(args.func(args))


if __name__ == "__main__":
    main()
