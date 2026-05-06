"""
build_final.py
==============
Merges all pipeline outputs into one clean final JSON.

Reads:
  - bn_compact.json          : id mapping (bn_00001 ↔ wiki_00000)
  - bn_marginal_cpds.jsonl   : joints + actual text + structure
  - bn_with_text.json        : generated text

For each BN produces:
  {
    "id":             "bn_00001",
    "source_id":      "wiki_00000",
    "title":          "...",
    "domain":         "...",
    "actual_text":    "...",
    "generated_text": "...",
    "nodes": {
      "financial system weakness": {
        "states": [...],
        "level":  0,
        "parents": {},
        "prior":  [0.20, 0.35, 0.40, 0.05]
      },
      "subprime mortgage crisis": {
        "states": ["active", "resolved", "none"],
        "level":  1,
        "parents": {
          "financial system weakness": {
            "parent_states": ["systemic fragility", "moderate weakness", "stable", "none"],
            "cpd_matrix": [
              [0.71, 0.48, 0.14, 0.05],   <- rows = child states
              [0.19, 0.32, 0.29, 0.16],   <- cols = parent states
              [0.10, 0.20, 0.57, 0.79]
            ]
          }
        }
      },
      "economic crisis severity": {
        "states": [...],
        "level":  2,
        "parents": {
          "bank lending capacity": {
            "parent_states": [...],
            "cpd_matrix": [[...], ...]    <- P(child | this parent) via Bayes
          },
          "financial system weakness": {
            "parent_states": [...],
            "cpd_matrix": [[...], ...]    <- P(child | this parent) via Bayes
          }
        }
      }
    }
  }

CPD matrix rules:
  - rows    = current node states
  - columns = parent states
  - NaN     = has_relationship=False OR computed value < 0.01

Pairwise P(child | parent_i) for multi-parent nodes is computed
from joint tables using Bayes: P(C|P) = P(C,P) / P(P)
where P(P=p) = sum over all child states of P(C,P=p).

Usage:
    python build_final.py
    python build_final.py --compact bn_compact.json
                          --marginal bn_marginal_cpds.jsonl
                          --generated bn_with_text.json
                          --output bn_final.json
"""

import os, json, argparse, math
from collections import defaultdict


FLOOR     = 0.01   # below this → NaN
NAN_TOKEN = None   # use JSON null for NaN (loads as None in Python)


# ─────────────────────────────────────────────────────────────────────────────
#  Level assignment from topological order
# ─────────────────────────────────────────────────────────────────────────────

def compute_levels(nodes: list, edges: list) -> dict:
    """
    Assign a level (layer) to each node.
    Level 0 = root nodes (no parents).
    Level n = max(parent levels) + 1.
    Matches the SVG DAG visualization layers.
    """
    in_edges = defaultdict(list)
    for e in edges:
        in_edges[e["child"]].append(e["parent"])

    level = {}

    def get_level(n):
        if n in level:
            return level[n]
        if not in_edges[n]:
            level[n] = 0
        else:
            level[n] = max(get_level(p) for p in in_edges[n]) + 1
        return level[n]

    for node in nodes:
        get_level(node["name"])

    return level


# ─────────────────────────────────────────────────────────────────────────────
#  Pairwise CPD recovery from joint table
# ─────────────────────────────────────────────────────────────────────────────

def recover_pairwise_cpd(joint_table: dict,
                          parent_states: list,
                          child_states: list,
                          has_rel: dict = None) -> list:
    """
    Recover P(child | parent) from joint table using Bayes:
      P(C=c | P=p) = P(C=c, P=p) / P(P=p)
      P(P=p) = sum_c P(C=c, P=p)

    Returns a matrix (list of lists):
      rows = child states
      cols = parent states
      NaN (None) if has_relationship=False OR value < FLOOR
    """
    # build matrix: row=child_state, col=parent_state
    matrix = []

    for cs in child_states:
        row = []
        for ps in parent_states:
            # check has_relationship flag
            no_rel = has_rel and not has_rel.get((ps, cs), True)

            # get joint value
            joint_val = float(
                joint_table.get(ps, {}).get(cs, 0.0)
            )

            # compute P(parent=ps) by marginalizing over child
            p_parent = sum(
                float(joint_table.get(ps, {}).get(c, 0.0))
                for c in child_states
            )

            if no_rel or p_parent < 1e-9:
                row.append(NAN_TOKEN)
                continue

            cond = joint_val / p_parent

            # apply NaN threshold
            if cond < FLOOR:
                row.append(NAN_TOKEN)
            else:
                row.append(round(cond, 4))

        matrix.append(row)

    return matrix


# ─────────────────────────────────────────────────────────────────────────────
#  Build node dict for one BN
# ─────────────────────────────────────────────────────────────────────────────

def build_node_dict(marginal_record: dict) -> dict:
    """
    Build the nodes dict from a bn_marginal_cpds record.
    Uses joint tables to recover pairwise P(child | parent_i).
    """
    nodes     = marginal_record["nodes"]
    edges     = marginal_record["edges"]
    joints    = marginal_record.get("joints", [])
    priors    = marginal_record.get("priors", [])
    raw_scores = marginal_record.get("raw_scores", [])

    # index joint tables: (parent, child) → table
    joint_index = {}
    for j in joints:
        joint_index[(j["parent"], j["child"])] = j["table"]

    # index raw scores for has_relationship:
    # (parent, parent_state, child) → has_relationship bool
    has_rel_index = {}
    for s in raw_scores:
        key = (s["parent"], s["parent_state"], s["child"])
        has_rel_index[key] = s.get("has_relationship", True)

    # index priors: node_name → distribution dict
    prior_index = {p["node"]: p["distribution"] for p in priors}

    # parent map: child → [parent names]
    parent_map = defaultdict(list)
    for e in edges:
        parent_map[e["child"]].append(e["parent"])

    # node state map
    node_state_map = {n["name"]: n["states"] for n in nodes}

    # compute levels
    levels = compute_levels(nodes, edges)

    result = {}

    for node in nodes:
        name        = node["name"]
        states      = node["states"]
        parent_names = parent_map.get(name, [])
        lv          = levels.get(name, 0)

        if not parent_names:
            # root node — use prior
            dist  = prior_index.get(name, {})
            prior = [round(float(dist.get(s, 1.0/len(states))), 4)
                     for s in states]
            result[name] = {
                "states":  states,
                "level":   lv,
                "parents": {},
                "prior":   prior,
            }

        else:
            # child node — one cpd matrix per parent
            parents_dict = {}

            for parent_name in parent_names:
                parent_states = node_state_map.get(parent_name, [])
                joint_table   = joint_index.get((parent_name, name), {})

                # build has_rel dict: (parent_state, child_state) → bool
                has_rel = {}
                for ps in parent_states:
                    for cs in states:
                        k = (parent_name, ps, name)
                        has_rel[(ps, cs)] = has_rel_index.get(k, True)

                matrix = recover_pairwise_cpd(
                    joint_table, parent_states, states, has_rel
                )

                parents_dict[parent_name] = {
                    "parent_states": parent_states,
                    "cpd_matrix":    matrix,
                }

            result[name] = {
                "states":  states,
                "level":   lv,
                "parents": parents_dict,
            }

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Merge pipeline outputs into one clean final JSON."
    )
    ap.add_argument("--compact",   default="bn_compact_updated.json",
                    help="Output of clean_bn.py (id mapping)")
    ap.add_argument("--marginal",  default="bn_marginal_cpds_updated.jsonl",
                    help="Output of cpd_calculator_marginal.py")
    ap.add_argument("--generated", default="bn_with_text_updated.json",
                    help="Output of bn_text_generator.py")
    ap.add_argument("--output",    default="bn_final_updated.json")
    args = ap.parse_args()

    # ── Load all three files ──────────────────────────────────────────────────
    for path in [args.compact, args.marginal, args.generated]:
        if not os.path.exists(path):
            print(f"ERROR: file not found: {path}")
            return

    print("Building final BN dataset")
    print(f"  Compact:   {args.compact}")
    print(f"  Marginal:  {args.marginal}")
    print(f"  Generated: {args.generated}")
    print(f"  Output:    {args.output}")
    print("-" * 55)

    # load compact — for id mapping
    compact_data = json.loads(open(args.compact).read())
    compact_bns  = compact_data["bayesian_networks"]

    # load marginal cpds — indexed by original id (wiki_00000)
    marginal_records = {}
    for line in open(args.marginal):
        if line.strip():
            rec = json.loads(line)
            marginal_records[rec["id"]] = rec

    marginal_ids = list(marginal_records.keys())

    # load generated text — indexed by new id (bn_00001)
    generated_data = json.loads(open(args.generated).read())
    generated_bns  = generated_data["bayesian_networks"]

    # build mapping: source_id (wiki_00000) → new_id (bn_00001)
    # fall back to positional matching if source_id is missing
    source_to_new = {}
    for idx, (new_id, bn) in enumerate(compact_bns.items()):
        source_id = bn.get("source_id", "")
        if source_id:
            source_to_new[source_id] = new_id
        elif idx < len(marginal_ids):
            source_to_new[marginal_ids[idx]] = new_id
            print(f"  [WARN] {new_id} has no source_id — positional match: {marginal_ids[idx]}")

    # ── Build final records ───────────────────────────────────────────────────
    final_bns = {}
    n_ok = n_skip = 0

    for idx, (new_id, compact_bn) in enumerate(compact_bns.items()):
        source_id = compact_bn.get("source_id", "")
        if not source_id and idx < len(marginal_ids):
            source_id = marginal_ids[idx]
        title = compact_bn.get("title", "")

        # get marginal record
        marginal_rec = marginal_records.get(source_id)
        if not marginal_rec:
            print(f"  SKIP {new_id} — no marginal record for {source_id}")
            n_skip += 1
            continue

        # get generated text
        gen_bn         = generated_bns.get(new_id, {})
        generated_text = gen_bn.get("text", "")
        actual_text    = marginal_rec.get("text", "")

        print(f"  [{n_ok+1}] {new_id} ← {source_id} | {title[:45]}")

        # build node dict with pairwise CPDs
        try:
            nodes_dict = build_node_dict(marginal_rec)
        except Exception as e:
            print(f"    ERROR building nodes: {e}")
            n_skip += 1
            continue

        final_bns[new_id] = {
            "id":             new_id,
            "source_id":      source_id,
            "title":          title,
            "domain":         compact_bn.get("domain", ""),
            "actual_text":    actual_text,
            "generated_text": generated_text,
            "nodes":          nodes_dict,
        }
        n_ok += 1

    # ── Write output ──────────────────────────────────────────────────────────
    output = {
        "metadata": {
            "total":   len(final_bns),
            "sources": {
                "compact":   args.compact,
                "marginal":  args.marginal,
                "generated": args.generated,
            }
        },
        "bayesian_networks": final_bns,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\n{'='*55}")
    print(f"Done.  OK={n_ok}  Skipped={n_skip}")
    print(f"Output: {args.output}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()