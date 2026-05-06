"""
clean_bn.py
===========
Reads bn_marginal_cpds.jsonl and writes a compact bn_compact.json
optimised for passing to the Claude API for text generation.

Keeps:
  - title, domain       (context for Claude's generation)
  - nodes               (name + states only)
  - edges               (parent → child only)
  - topological_order   (traversal order)
  - cpds                (rounded to 2dp, derivation dropped)

Drops everything else:
  - text, url           (source article)
  - marginals, joints   (intermediate computation)
  - raw_scores          (pre-normalization values)
  - svg_path            (file reference)
  - cpd_issues          (validation metadata)
  - real, n_nodes, n_edges, source_id
  - derivation          (internal note inside each CPD)

CPD cleaning:
  - All float values rounded to 2 decimal places
  - derivation key removed from every CPD entry
  - Structure preserved exactly (prior / matrix / nested dict)

Usage:
    python clean_bn.py
    python clean_bn.py --input bn_marginal_cpds.jsonl --output bn_compact.json
"""

import os, json, argparse


# ─────────────────────────────────────────────────────────────────────────────
#  CPD cleaning helpers
# ─────────────────────────────────────────────────────────────────────────────

def round_recursive(obj, dp=2):
    """Recursively round all floats in any nested structure."""
    if isinstance(obj, float):
        return round(obj, dp)
    if isinstance(obj, list):
        return [round_recursive(v, dp) for v in obj]
    if isinstance(obj, dict):
        return {k: round_recursive(v, dp) for k, v in obj.items()}
    return obj


def clean_cpd_entry(cpd: dict, dp: int = 2) -> dict:
    """
    Clean a single CPD entry:
      - Remove 'derivation' key
      - Round all float values to dp decimal places
      - Keep structure: prior / matrix / nested cpd dict
    """
    cleaned = {}

    for key, val in cpd.items():
        if key == "derivation":
            continue                      # drop this
        cleaned[key] = round_recursive(val, dp)

    return cleaned


def clean_cpds(cpds: list, dp: int = 2) -> list:
    return [clean_cpd_entry(c, dp) for c in cpds]


# ─────────────────────────────────────────────────────────────────────────────
#  Record cleaning
# ─────────────────────────────────────────────────────────────────────────────

def clean_record(record: dict, new_id: str, dp: int = 2) -> dict:
    """
    Extract and clean a single BN record.
    Keeps only what Claude needs for text generation.
    """
    nodes = [
        {"name": n["name"], "states": n["states"]}
        for n in record.get("nodes", [])
    ]

    edges = [
        {"parent": e["parent"], "child": e["child"]}
        for e in record.get("edges", [])
    ]

    cpds = clean_cpds(record.get("cpds", []), dp)

    return {
        "id":                new_id,
        "source_id":         record.get("id", ""),
        "title":             record.get("title", ""),
        "domain":            record.get("domain", ""),
        "nodes":             nodes,
        "edges":             edges,
        "topological_order": record.get("topological_order", []),
        "cpds":              cpds,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Clean and compact BN records for Claude API text generation."
    )
    ap.add_argument("--input",  default="bn_marginal_cpds_updated.jsonl",
                    help="Input jsonl (default: bn_marginal_cpds_updated.jsonl)")
    ap.add_argument("--output", default="bn_compact_updated.json",
                    help="Output json (default: bn_compact_updated.json)")
    ap.add_argument("--dp",     type=int, default=2,
                    help="Decimal places for CPD rounding (default: 2)")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: input not found: {args.input}")
        return

    records = [json.loads(l) for l in open(args.input) if l.strip()]
    print(f"Clean BN")
    print(f"  Input:   {args.input}  ({len(records)} records)")
    print(f"  Output:  {args.output}")
    print(f"  Rounding CPDs to {args.dp} decimal places")
    print("-" * 55)

    bayesian_networks = {}

    for i, record in enumerate(records):
        new_id  = f"bn_{i+1:05d}"
        cleaned = clean_record(record, new_id, args.dp)
        bayesian_networks[new_id] = cleaned

        n_nodes = len(cleaned["nodes"])
        n_edges = len(cleaned["edges"])
        n_cpds  = len(cleaned["cpds"])
        print(f"  [{i+1}/{len(records)}] {new_id} | "
              f"nodes={n_nodes} edges={n_edges} cpds={n_cpds} | "
              f"{record.get('title','')[:40]}")

    output = {
        "metadata": {
            "total":    len(bayesian_networks),
            "source":   args.input,
            "cpd_dp":   args.dp,
        },
        "bayesian_networks": bayesian_networks,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, separators=(',', ':'))

    # ── size report ───────────────────────────────────────────────────────────
    size_kb = os.path.getsize(args.output) / 1024
    avg_kb  = size_kb / max(len(bayesian_networks), 1)
    print(f"\nDone.")
    print(f"  Total BNs:       {len(bayesian_networks)}")
    print(f"  Output size:     {size_kb:.1f} KB")
    print(f"  Avg per BN:      {avg_kb:.1f} KB")
    print(f"  Output:          {args.output}")


if __name__ == "__main__":
    main()