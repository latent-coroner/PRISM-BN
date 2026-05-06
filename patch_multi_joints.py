"""
patch_multi_joints.py
=====================
Post-processes bn_marginal_raw.jsonl to add exact full multi-parent joints
(Pass 5) for nodes with 2-3 parents.

Reads:  bn_marginal_raw.jsonl
Writes: bn_marginal_raw_v2.jsonl  (drop-in replacement)

Only re-runs API calls for records that have nodes with 2-3 parents.
Records with no multi-parent nodes are copied through unchanged.
Nodes with 4+ parents keep the Naive Bayes fallback in cpd_calculator.py.

Usage:
    python patch_multi_joints.py
    python patch_multi_joints.py --input bn_marginal_raw.jsonl --output bn_marginal_raw_v2.jsonl
    python patch_multi_joints.py --resume   # skip already-written records
"""

import os, json, time, argparse
from collections import defaultdict
import anthropic
from extractor import PASS5_SYSTEM, pass5_prompt, call_claude

FLOOR = 0.0001


# ─────────────────────────────────────────────────────────────────────────────
#  Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_multi_joints(joints_raw: list, nodes: list, edges: list) -> list:
    node_map   = {n["name"]: n["states"] for n in nodes}
    parent_map = defaultdict(list)
    for e in edges:
        parent_map[e["child"]].append(e["parent"])

    validated = []
    for j in joints_raw:
        child = j.get("child")
        table = j.get("table", {})
        if not child or not table:
            continue

        child_states = node_map.get(child, [])
        parent_names = parent_map.get(child, [])
        if not child_states or len(parent_names) < 2 or len(parent_names) > 3:
            continue

        parents_info = [{"name": pn, "states": node_map.get(pn, [])}
                        for pn in parent_names]

        def collect_leaves(subtable, depth):
            if depth == len(parents_info):
                return [max(FLOOR, float(subtable.get(cs, FLOOR)))
                        for cs in child_states]
            p      = parents_info[depth]
            values = []
            for ps in p["states"]:
                k   = f"{p['name']}={ps}"
                values.extend(collect_leaves(subtable.get(k, {}), depth + 1))
            return values

        leaves = collect_leaves(table, 0)
        total  = sum(leaves)
        if total == 0:
            continue

        # rebuild normalized table
        leaf_iter = iter(v / total for v in leaves)

        def build_norm(subtable, depth):
            if depth == len(parents_info):
                return {cs: round(next(leaf_iter), 6) for cs in child_states}
            p      = parents_info[depth]
            result = {}
            for ps in p["states"]:
                k         = f"{p['name']}={ps}"
                result[k] = build_norm(subtable.get(k, {}), depth + 1)
            return result

        norm_table = build_norm(table, 0)
        validated.append({
            "child":   child,
            "parents": parent_names,
            "table":   norm_table,
        })

    return validated


# ─────────────────────────────────────────────────────────────────────────────
#  Per-record patch
# ─────────────────────────────────────────────────────────────────────────────

def patch_record(record: dict, client, model: str) -> list:
    nodes = record["nodes"]
    edges = record["edges"]
    text  = record.get("text", "")

    node_map   = {n["name"]: n["states"] for n in nodes}
    parent_map = defaultdict(list)
    for e in edges:
        parent_map[e["child"]].append(e["parent"])

    targets = [child for child, pnames in parent_map.items()
               if 2 <= len(pnames) <= 3]
    if not targets:
        return []

    # Estimate token need per node (~25 tokens per leaf value)
    def node_combos(child):
        pnames  = parent_map[child]
        n_combo = len(node_map.get(child, []))
        for pn in pnames:
            n_combo *= len(node_map.get(pn, []))
        return n_combo

    # Split targets so no chunk exceeds ~5000 estimated tokens
    TOKEN_PER_COMBO = 25
    TOKEN_BUDGET    = 5000
    chunks = []
    current, current_tok = [], 0
    for child in targets:
        est = node_combos(child) * TOKEN_PER_COMBO
        if current and current_tok + est > TOKEN_BUDGET:
            chunks.append(current)
            current, current_tok = [], 0
        current.append(child)
        current_tok += est
    if current:
        chunks.append(current)

    all_joints = []
    for i, chunk in enumerate(chunks):
        prompt = pass5_prompt(nodes, edges, text, target_children=chunk)
        if not prompt:
            continue

        n_combos = sum(node_combos(child) for child in chunk)
        max_tok = max(2000, min(8000, n_combos * TOKEN_PER_COMBO))

        result = call_claude(client, PASS5_SYSTEM, prompt, model,
                             max_tokens=max_tok)
        if result and "joints" in result:
            validated = validate_multi_joints(result["joints"], nodes, edges)
            all_joints.extend(validated)
            print(f"        chunk {i+1}/{len(chunks)}: "
                  f"{len(validated)}/{len(chunk)} joints OK")
        else:
            print(f"        chunk {i+1}/{len(chunks)}: FAILED")

    return all_joints


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Add exact multi-parent joints (Pass 5) to BN records."
    )
    ap.add_argument("--input",      default="bn_marginal_raw.jsonl")
    ap.add_argument("--output",     default="bn_marginal_raw_updated.jsonl")
    ap.add_argument("--api-key",    default=None)
    ap.add_argument("--model",      default=os.getenv("EXTRACTION_MODEL", "extraction-model"))
    ap.add_argument("--chunk-size", type=int,   default=3,
                    help="Multi-parent child nodes per API call (default: 3)")
    ap.add_argument("--delay",      type=float, default=0.5)
    ap.add_argument("--resume",     action="store_true",
                    help="Skip records already present in output file")
    args = ap.parse_args()

    key = args.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        print("ERROR: provide --api-key or set ANTHROPIC_API_KEY")
        return

    if not os.path.exists(args.input):
        print(f"ERROR: input not found: {args.input}")
        return

    client  = anthropic.Anthropic(api_key=key)
    records = [json.loads(l) for l in open(args.input) if l.strip()]

    done_ids = set()
    if args.resume and os.path.exists(args.output):
        for line in open(args.output):
            done_ids.add(json.loads(line).get("id", ""))
        print(f"Resuming — {len(done_ids)} records already written")

    print(f"Pass 5 — Multi-parent joint patch")
    print(f"  Input:      {args.input}  ({len(records)} records)")
    print(f"  Output:     {args.output}")
    print(f"  Model:      {args.model}")
    print(f"  Chunk size: {args.chunk_size} nodes/call")
    print("-" * 55)

    n_patched = n_passthrough = n_error = 0

    with open(args.output, "a" if args.resume else "w") as out:
        for i, record in enumerate(records):
            rid   = record.get("id", f"record_{i:05d}")
            title = record.get("title", rid)[:50]

            if rid in done_ids:
                continue

            print(f"  [{i+1}/{len(records)}] {title}")

            try:
                multi_joints = patch_raecord(record, client, args.model)
                if multi_joints:
                    print(f"        → {len(multi_joints)} multi-parent joints added")
                    n_patched += 1
                else:
                    n_passthrough += 1

                out.write(json.dumps({**record, "multi_joints": multi_joints}) + "\n")
                out.flush()

            except Exception as e:
                print(f"        ERROR: {e}")
                n_error += 1

            time.sleep(args.delay)

    print(f"\n{'='*55}")
    print(f"Done.  Patched={n_patched}  Pass-through={n_passthrough}  Errors={n_error}")
    print(f"Output: {args.output}")
    print(f"\nNext step: rerun cpd_calculator.py with --input {args.output}")


if __name__ == "__main__":
    main()
