"""
cpd_calculator_marginal.py
==========================
Reads bn_marginal_raw.jsonl (output of bn_extractor_marginal.py) and
recovers CPDs using Bayes' theorem:

  P(child | parent) = P(child, parent) / P(parent)

Where:
  P(child, parent)  — joint probability from Pass 4
  P(parent)         — marginal probability from Pass 3

Also generates one SVG per BN (DAG only).
Output: bn_marginal_cpds.jsonl

The math in detail
------------------
For each edge parent → child and each parent state ps:

  P(child=cs | parent=ps) = P(child=cs, parent=ps) / P(parent=ps)

  P(parent=ps) is computed by summing the joint table over all child states:
    P(parent=ps) = Σ_cs  P(child=cs, parent=ps)

  This is called "marginalizing out" the child variable.

  If the recovered marginal P(parent=ps) is very small (< 0.001),
  we fall back to a uniform distribution to avoid division instability.

CPD matrix format  (same as cpd_calculator.py):
  - rows    = current node states
  - columns = parent states  (single parent)
  - nested  = parent1_state → parent2_state → row  (multi-parent)

Usage:
    python cpd_calculator_marginal.py
    python cpd_calculator_marginal.py --input bn_marginal_raw.jsonl
    python cpd_calculator_marginal.py --svg-dir ./svgs_marginal --strict
"""

import os, json, argparse
from collections import defaultdict


FLOOR = 0.001


# ─────────────────────────────────────────────────────────────────────────────
#  Core Bayes recovery
# ─────────────────────────────────────────────────────────────────────────────

def recover_conditional(joint_table: dict, parent_states: list,
                        child_states: list) -> dict:
    """
    Recover P(child | parent) from a joint table using Bayes' theorem.

    joint_table[parent_state][child_state] = P(parent=ps, child=cs)

    Returns:
      conditional[parent_state][child_state] = P(child=cs | parent=ps)

    Each row (per parent_state) sums to 1.0.
    """
    conditional = {}

    for ps in parent_states:
        row = joint_table.get(ps, {})

        # Step 1 — compute P(parent=ps) by marginalizing out child
        # P(parent=ps) = Σ_cs P(parent=ps, child=cs)
        p_parent = sum(float(row.get(cs, 0.0)) for cs in child_states)

        if p_parent < FLOOR:
            # Parent state is essentially impossible — use uniform fallback
            u = round(1.0 / len(child_states), 6)
            conditional[ps] = {cs: u for cs in child_states}
            continue

        # Step 2 — divide each joint by P(parent=ps)
        # P(child=cs | parent=ps) = P(parent=ps, child=cs) / P(parent=ps)
        cond_row = {}
        for cs in child_states:
            joint_val  = max(0.0, float(row.get(cs, 0.0)))
            cond_row[cs] = round(joint_val / p_parent, 6)

        # Step 3 — fix rounding drift so row sums to exactly 1.0
        total = sum(cond_row.values())
        if abs(total - 1.0) > 0.001:
            cond_row = {cs: round(v / total, 6) for cs, v in cond_row.items()}
        diff = round(1.0 - sum(cond_row.values()), 6)
        if diff != 0:
            largest = max(cond_row, key=cond_row.__getitem__)
            cond_row[largest] = round(cond_row[largest] + diff, 6)

        conditional[ps] = cond_row

    return conditional


# ─────────────────────────────────────────────────────────────────────────────
#  CPD builders
# ─────────────────────────────────────────────────────────────────────────────

def build_prior_cpd(node: dict, marginals: dict) -> dict:
    """
    Root node — use marginal directly as prior.

    {
      "node":        "economic shock",
      "node_states": ["recession", "inflation", "none"],
      "parents":     [],
      "prior":       [0.35, 0.40, 0.25]
    }
    """
    states = node["states"]
    dist   = marginals.get(node["name"], {})

    if dist:
        probs = [max(FLOOR, float(dist.get(s, FLOOR))) for s in states]
        total = sum(probs)
        probs = [round(v / total, 6) for v in probs]
        diff  = round(1.0 - sum(probs), 6)
        if diff != 0:
            probs[probs.index(max(probs))] += diff
    else:
        u     = round(1.0 / len(states), 6)
        probs = [u] * len(states)

    return {
        "node":        node["name"],
        "node_states": states,
        "parents":     [],
        "prior":       probs,
    }


def build_cpd_single_parent(node: dict, parent: dict,
                             joint_index: dict) -> dict:
    """
    Single-parent node — recover P(child|parent) via Bayes.
    Matrix: rows = node states, columns = parent states.

    {
      "node":          "social outcome",
      "node_states":   ["unemployment", "poverty", "none"],
      "parent":        "economic shock",
      "parent_states": ["recession", "inflation", "none"],
      "matrix": [
        [0.52, 0.30, 0.01],   ← unemployment row
        [0.41, 0.22, 0.01],   ← poverty row
        [0.07, 0.48, 0.98]    ← none row
      ],
      "derivation": "P(child|parent) = P(child,parent) / P(parent)"
    }
    """
    node_states   = node["states"]
    parent_name   = parent["name"]
    parent_states = parent["states"]

    key        = (parent_name, node["name"])
    joint_data = joint_index.get(key, {})

    # recover conditional from joint
    conditional = recover_conditional(joint_data, parent_states, node_states)

    # build matrix: row = node_state, col = parent_state
    matrix = [
        [conditional.get(ps, {}).get(ns, round(1.0/len(node_states), 6))
         for ps in parent_states]
        for ns in node_states
    ]

    return {
        "node":          node["name"],
        "node_states":   node_states,
        "parent":        parent_name,
        "parent_states": parent_states,
        "matrix":        matrix,
        "derivation":    "P(child|parent) = P(child,parent) / P(parent)",
    }


def build_cpd_multi_parent(node: dict, parents: list,
                            joint_index: dict,
                            marginals: dict) -> dict:
    """
    Multi-parent node — chain rule approximation.

    For a node C with parents A and B, we approximate using the
    chain rule and available pairwise joints:

      P(C=c | A=a, B=b) ∝ P(C=c | A=a) × P(C=c | B=b) / P(C=c)

    This is the Naive Bayes approximation — assumes parents are
    conditionally independent given the child. It's an approximation
    but is standard when full joint data is not available.

    {
      "node":       "long-term impact",
      "node_states":["stagnation", "recovery", "none"],
      "parents": [
        {"name": "economic shock",  "states": [...]},
        {"name": "policy response", "states": [...]}
      ],
      "cpd": {
        "economic shock=recession": {
          "policy response=austerity": [0.68, 0.22, 0.10],
          ...
        }
      },
      "derivation": "Naive Bayes: P(C|A,B) ∝ P(C|A)×P(C|B)/P(C)"
    }
    """
    node_states = node["states"]
    node_name   = node["name"]

    # get P(C=c) from marginals for the denominator
    node_marginal = marginals.get(node_name, {})
    p_child = [max(FLOOR, float(node_marginal.get(cs, FLOOR)))
               for cs in node_states]
    p_child_total = sum(p_child)
    p_child = [v / p_child_total for v in p_child]

    # recover P(C | each parent) separately
    pairwise = {}
    for p in parents:
        key        = (p["name"], node_name)
        joint_data = joint_index.get(key, {})
        pairwise[p["name"]] = recover_conditional(
            joint_data, p["states"], node_states
        )

    def build_nested(parent_idx: int, combo_so_far: list):
        if parent_idx == len(parents):
            # leaf — apply Naive Bayes approximation
            # P(C=c | A=a, B=b) ∝ P(C=c|A=a) × P(C=c|B=b) / P(C=c)
            scores = []
            for ci, cs in enumerate(node_states):
                score = 1.0
                for p, ps in zip(parents, combo_so_far):
                    cond_val = pairwise[p["name"]].get(ps, {}).get(cs, FLOOR)
                    score   *= max(FLOOR, cond_val)
                # divide by P(C=c) to avoid double-counting the prior
                score = score / max(FLOOR, p_child[ci])
                scores.append(max(FLOOR, score))

            # normalize
            total = sum(scores)
            row   = [round(s / total, 6) for s in scores]
            diff  = round(1.0 - sum(row), 6)
            if diff != 0:
                row[row.index(max(row))] += diff
            return row

        p      = parents[parent_idx]
        result = {}
        for ps in p["states"]:
            k         = f"{p['name']}={ps}"
            result[k] = build_nested(parent_idx + 1, combo_so_far + [ps])
        return result

    return {
        "node":        node_name,
        "node_states": node_states,
        "parents":     [{"name": p["name"], "states": p["states"]}
                        for p in parents],
        "cpd":         build_nested(0, []),
        "derivation":  "Naive Bayes: P(C|A,B) ∝ P(C|A)×P(C|B)/P(C)",
    }


def build_cpd_exact_multi_parent(node: dict, parents: list,
                                  full_joint: dict) -> dict:
    """
    Exact CPD from full multi-parent joint table (from Pass 5).

    P(C=c | A=a, B=b) = P(C=c, A=a, B=b) / P(A=a, B=b)
    P(A=a, B=b)        = Σ_c P(C=c, A=a, B=b)

    full_joint["table"] is nested: "A=a" → "B=b" → {child_state: joint_prob}
    All leaf values in the table sum to 1.0 (full joint).
    """
    node_states = node["states"]
    table       = full_joint.get("table", {})

    def compute_leaf(leaf: dict) -> list:
        total = sum(max(0.0, float(leaf.get(cs, 0.0))) for cs in node_states)
        if total < FLOOR:
            u = round(1.0 / len(node_states), 6)
            return [u] * len(node_states)
        row  = [round(max(0.0, float(leaf.get(cs, 0.0))) / total, 6)
                for cs in node_states]
        diff = round(1.0 - sum(row), 6)
        if diff != 0:
            row[row.index(max(row))] = round(row[row.index(max(row))] + diff, 6)
        return row

    def build_nested(subtable: dict, parent_idx: int):
        if parent_idx == len(parents):
            return compute_leaf(subtable)
        p      = parents[parent_idx]
        result = {}
        for ps in p["states"]:
            k         = f"{p['name']}={ps}"
            result[k] = build_nested(subtable.get(k, {}), parent_idx + 1)
        return result

    return {
        "node":        node["name"],
        "node_states": node_states,
        "parents":     [{"name": p["name"], "states": p["states"]}
                        for p in parents],
        "cpd":         build_nested(table, 0),
        "derivation":  "Exact: P(C|parents) = P(C,parents) / P(parents)",
    }


def build_all_cpds(record: dict) -> list:
    nodes        = record["nodes"]
    edges        = record["edges"]
    marginals    = record.get("marginals", {})
    joints       = record.get("joints", [])
    multi_joints = record.get("multi_joints", [])

    node_map = {n["name"]: n for n in nodes}

    # pairwise joint index: (parent_name, child_name) → table
    joint_index = {}
    for j in joints:
        key = (j["parent"], j["child"])
        joint_index[key] = j["table"]

    # full multi-parent joint index: child_name → joint entry
    multi_joint_index = {mj["child"]: mj for mj in multi_joints}

    # parent map: child_name → [parent node dicts]
    parent_map = defaultdict(list)
    for e in edges:
        parent_map[e["child"]].append(node_map[e["parent"]])

    cpds = []
    for node in nodes:
        parents = parent_map.get(node["name"], [])
        if not parents:
            cpds.append(build_prior_cpd(node, marginals))
        elif len(parents) == 1:
            cpds.append(build_cpd_single_parent(node, parents[0], joint_index))
        elif node["name"] in multi_joint_index:
            cpds.append(build_cpd_exact_multi_parent(
                node, parents, multi_joint_index[node["name"]]))
        else:
            cpds.append(build_cpd_multi_parent(node, parents,
                                                joint_index, marginals))
    return cpds


# ─────────────────────────────────────────────────────────────────────────────
#  Verification
# ─────────────────────────────────────────────────────────────────────────────

def verify_cpds(cpds: list) -> list:
    issues = []
    for cpd in cpds:
        name = cpd["node"]

        if not cpd.get("parents") and "prior" in cpd:
            total = sum(cpd["prior"])
            if abs(total - 1.0) > 0.01:
                issues.append(f"{name} prior sums to {total:.4f}")
            continue

        if "matrix" in cpd:
            node_states   = cpd["node_states"]
            parent_states = cpd["parent_states"]
            matrix        = cpd["matrix"]
            for col_idx, ps in enumerate(parent_states):
                col_sum = sum(matrix[r][col_idx] for r in range(len(node_states)))
                if abs(col_sum - 1.0) > 0.01:
                    issues.append(
                        f"{name}|{cpd['parent']}={ps} col sums to {col_sum:.4f}"
                    )

        elif "cpd" in cpd:
            def check_nested(d, path=""):
                if isinstance(d, list):
                    total = sum(d)
                    if abs(total - 1.0) > 0.01:
                        issues.append(f"{name}|{path} row sums to {total:.4f}")
                else:
                    for k, v in d.items():
                        check_nested(v, f"{path} {k}".strip())
            check_nested(cpd["cpd"])

    return issues


# ─────────────────────────────────────────────────────────────────────────────
#  SVG  (identical logic to cpd_calculator.py)
# ─────────────────────────────────────────────────────────────────────────────

NODE_W = 160; NODE_H = 48; H_GAP = 60; V_GAP = 90; FONT_SIZE = 13; PADDING = 40

COLORS = {
    "root":  {"fill": "#E8734A", "stroke": "#A84A28", "text": "#FFFFFF"},
    "child": {"fill": "#F5F5F5", "stroke": "#999999", "text": "#333333"},
    "edge":  "#666666", "bg": "#FFFFFF",
}

def _topo_layers(nodes, edges):
    in_edges = defaultdict(list)
    for e in edges:
        in_edges[e["child"]].append(e["parent"])
    layer = {}
    def get_layer(n):
        if n in layer:
            return layer[n]
        layer[n] = 0 if not in_edges[n] else max(get_layer(p) for p in in_edges[n]) + 1
        return layer[n]
    for n in [nd["name"] for nd in nodes]:
        get_layer(n)
    max_l  = max(layer.values()) if layer else 0
    layers = [[] for _ in range(max_l + 1)]
    for n in [nd["name"] for nd in nodes]:
        layers[layer[n]].append(n)
    return layers

def _wrap(text, max_chars=18):
    if len(text) <= max_chars:
        return [text]
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > max_chars:
            lines.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return lines[:2]

def _x(s):
    return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace('"',"&quot;")

def generate_svg(record: dict) -> str:
    nodes     = record["nodes"]
    edges     = record["edges"]
    title     = record.get("title", record.get("id", "BN"))
    child_set = {e["child"] for e in edges}

    if not nodes:
        return '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="60"><text x="10" y="30">No nodes</text></svg>'

    layers = _topo_layers(nodes, edges)
    max_layer_w = max(len(l)*NODE_W + max(0,len(l)-1)*H_GAP for l in layers)
    canvas_w    = max_layer_w + 2*PADDING
    canvas_h    = len(layers)*(NODE_H+V_GAP) + PADDING + 40

    node_pos = {}
    for li, layer in enumerate(layers):
        n       = len(layer)
        total_w = n*NODE_W + max(0,n-1)*H_GAP
        x_start = (canvas_w - total_w) // 2
        y       = 50 + li*(NODE_H+V_GAP) + NODE_H//2
        for i, name in enumerate(layer):
            node_pos[name] = (x_start + i*(NODE_W+H_GAP) + NODE_W//2, y)

    parts = [
        '<defs><marker id="arr" markerWidth="10" markerHeight="7" '
        f'refX="9" refY="3.5" orient="auto"><polygon points="0 0,10 3.5,0 7" fill="{COLORS["edge"]}"/>'
        '</marker></defs>',
        f'<rect width="{canvas_w}" height="{canvas_h}" fill="{COLORS["bg"]}" rx="8"/>',
        f'<text x="{canvas_w//2}" y="24" text-anchor="middle" '
        f'font-family="Arial,sans-serif" font-size="15" font-weight="bold" '
        f'fill="#333">{_x(title[:60])}</text>',
    ]

    for e in edges:
        s, d = e["parent"], e["child"]
        if s not in node_pos or d not in node_pos: continue
        x1,y1 = node_pos[s]; x2,y2 = node_pos[d]
        parts.append(
            f'<line x1="{x1}" y1="{y1+NODE_H//2}" x2="{x2}" y2="{y2-NODE_H//2}" '
            f'stroke="{COLORS["edge"]}" stroke-width="1.8" marker-end="url(#arr)"/>'
        )

    for node in nodes:
        name = node["name"]
        if name not in node_pos: continue
        cx,cy  = node_pos[name]
        x,y    = cx-NODE_W//2, cy-NODE_H//2
        is_root = name not in child_set
        c       = COLORS["root"] if is_root else COLORS["child"]
        parts.append(
            f'<rect x="{x}" y="{y}" width="{NODE_W}" height="{NODE_H}" '
            f'rx="8" fill="{c["fill"]}" stroke="{c["stroke"]}" stroke-width="1.5"/>'
        )
        lines = _wrap(name)
        for li2, line in enumerate(lines):
            offset = (li2 - (len(lines)-1)/2) * 16
            parts.append(
                f'<text x="{cx}" y="{cy+offset}" text-anchor="middle" '
                f'dominant-baseline="middle" font-family="Arial,sans-serif" '
                f'font-size="{FONT_SIZE}" fill="{c["text"]}">{_x(line)}</text>'
            )

    ly = canvas_h - 24
    parts += [
        f'<rect x="10" y="{ly}" width="14" height="14" rx="3" fill="{COLORS["root"]["fill"]}"/>',
        f'<text x="28" y="{ly+11}" font-family="Arial,sans-serif" font-size="11" fill="#555">Root node</text>',
        f'<rect x="100" y="{ly}" width="14" height="14" rx="3" fill="{COLORS["child"]["fill"]}" stroke="{COLORS["child"]["stroke"]}"/>',
        f'<text x="118" y="{ly+11}" font-family="Arial,sans-serif" font-size="11" fill="#555">Child node</text>',
    ]

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_w}" '
        f'height="{canvas_h}" viewBox="0 0 {canvas_w} {canvas_h}">\n'
        + "\n".join(parts) + "\n</svg>"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Recover CPDs via Bayes from bn_marginal_raw.jsonl"
    )
    ap.add_argument("--input",   default="bn_marginal_raw_updated.jsonl")
    ap.add_argument("--output",  default="bn_marginal_cpds_updated.jsonl")
    ap.add_argument("--svg-dir", default="svgs_marginal")
    ap.add_argument("--strict",  action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: input not found: {args.input}")
        return

    os.makedirs(args.svg_dir, exist_ok=True)
    records = [json.loads(l) for l in open(args.input) if l.strip()]

    print(f"CPD Calculator  (marginal / Bayes approach)")
    print(f"  Input:   {args.input}  ({len(records)} records)")
    print(f"  Output:  {args.output}")
    print(f"  SVGs:    {args.svg_dir}/")
    print("-" * 55)

    n_ok = n_warn = n_skip = 0

    with open(args.output, "w") as out:
        for i, record in enumerate(records):
            rid   = record.get("id", f"record_{i:05d}")
            title = record.get("title", rid)[:50]

            try:
                cpds   = build_all_cpds(record)
                issues = verify_cpds(cpds)

                if issues and args.strict:
                    print(f"  [{i+1}] SKIP  {title}")
                    for iss in issues[:3]:
                        print(f"        ↳ {iss}")
                    n_skip += 1
                    continue

                if issues:
                    print(f"  [{i+1}] WARN  {title}  ({len(issues)} issues)")
                    n_warn += 1
                else:
                    print(f"  [{i+1}] OK    {title}")
                    n_ok += 1

                svg_str  = generate_svg(record)
                svg_path = os.path.join(args.svg_dir, f"{rid}.svg")
                with open(svg_path, "w", encoding="utf-8") as sf:
                    sf.write(svg_str)

                out.write(json.dumps({
                    **record,
                    "cpds":       cpds,
                    "cpd_issues": issues,
                    "svg_path":   svg_path,
                }) + "\n")

            except Exception as e:
                print(f"  [{i+1}] ERROR {title}: {e}")
                n_skip += 1

    print(f"\n{'='*55}")
    print(f"Done.  OK={n_ok}  Warnings={n_warn}  Skipped={n_skip}")
    print(f"Output: {args.output}  |  SVGs: {args.svg_dir}/")


if __name__ == "__main__":
    main()