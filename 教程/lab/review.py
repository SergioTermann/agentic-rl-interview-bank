"""Compare two fixed policies on paired tasks from train.py --cases."""

import argparse
import csv
import random
from pathlib import Path


def compare(path, split, seed, base, candidate):
    by_method = {base: {}, candidate: {}}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["method"] in by_method and row["split"] == split and int(row["seed"]) == seed:
                by_method[row["method"]][int(row["task_id"])] = row
    a, b = by_method[base], by_method[candidate]
    if not a or a.keys() != b.keys():
        raise ValueError("Both methods must cover the same nonempty task set.")
    deltas = [int(b[i]["verified"]) - int(a[i]["verified"]) for i in a]
    rng = random.Random(310)
    n = len(deltas)
    boot = sorted(sum(rng.choices(deltas, k=n)) / n for _ in range(2000))
    print(f"n={n} wins={deltas.count(1)} losses={deltas.count(-1)} "
          f"delta={sum(deltas)/n:.4f} paired task bootstrap 95%=[{boot[49]:.4f}, {boot[1949]:.4f}]")
    failures = [b[i] for i in a if not int(b[i]["verified"])]
    for row in failures[:3]:
        print("failed trace:", row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("--split", choices=["dev", "test", "shift"], default="dev")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--base", default="sft")
    parser.add_argument("--candidate", default="sft-grpo")
    args = parser.parse_args()
    compare(args.cases, args.split, args.seed, args.base, args.candidate)
