#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from checkpoint_dag_diffusion.properties import rerank_property_records


def _targets(values: list[str]) -> dict[str, float]:
    targets: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=VALUE, got {value!r}.")
        name, target = value.split("=", 1)
        targets[name] = float(target)
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerank CCD candidates by independent scalar-property evaluation.")
    parser.add_argument("--samples_json", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--target", action="append", required=True, help="Property target, e.g. logP=1.5")
    parser.add_argument("--num_select", type=int, required=True)
    args = parser.parse_args()
    records = json.loads(Path(args.samples_json).read_text(encoding="utf-8"))
    selected = rerank_property_records(records, targets=_targets(args.target), num_select=args.num_select)
    Path(args.out).write_text(json.dumps(selected, indent=2), encoding="utf-8")
    print(json.dumps({"candidates": len(records), "selected": len(selected), "out": args.out}, indent=2))


if __name__ == "__main__":
    main()
