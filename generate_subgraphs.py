"""
generate_subgraphs.py
=====================
Reads bn_final.json and bn_marginal_cpds.jsonl to generate valid
connected subgraphs from each Bayesian Network.

Algorithm:
  Step 1 — Enumerate all valid connected subgraphs within a size range
           using BFS, DFS, or both traversals from every possible start node.
           Report count to user.
  Step 2 — Sample N subgraphs (N must be ≤ possible count).
  Step 3 — Induce edges: keep only edges where both endpoints are
           in the node subset.
  Step 4 — Re-root cut nodes: nodes that lost all parents become
           new roots. Prior = marginal from bn_marginal_cpds.jsonl.
  Step 5 — Recompute CPDs for multi-parent nodes with surviving
           parents using marginals + joints via Bayes:
           P(child|parent_i) = P(child, parent_i) / P(parent_i)

Output: bn_subgraphs.json
  Same structure as bn_final.json.
  Subgraph IDs: bn_00001_sub_001, bn_00001_sub_002, ...
  actual_text and generated_text carried over from parent BN.

Usage:
    # fixed size with BFS (backward-compatible)
    python generate_subgraphs.py --count-only --target-size 5
    python generate_subgraphs.py --target-size 5 --n-subgraphs 3

    # variable size range
    python generate_subgraphs.py --min-size 3 --max-size 7 --n-subgraphs 5

    # traversal options: bfs, dfs, or both (combined + deduplicated)
    python generate_subgraphs.py --min-size 3 --max-size 6 --n-subgraphs 4 --traversal both
    python generate_subgraphs.py --target-size 5 --n-subgraphs 3 --traversal dfs
"""

import os, json, random, argparse
from collections import defaultdict, deque


FLOOR = 0.001


# ─────────────────────────────────────────────────────────────────────────────
#  Load marginal records
# ─────────────────────────────────────────────────────────────────────────────

def load_marginal_records(path: str) -> dict:
    """
    Load bn_marginal_cpds.jsonl indexed by original id (wiki_00000).
    Returns: {wiki_id: record}
    """
    records = {}
    for line in open(path):
        if line.strip():
            rec = json.loads(line)
            records[rec["id"]] = rec
    return records


def get_marginals(marginal_rec: dict) -> dict:
    """
    Extract marginals dict from marginal record.
    Returns: {node_name: {state: prob}}
    """
    return marginal_rec.get("marginals", {})


def get_joint_index(marginal_rec: dict) -> dict:
    """
    Build joint index from marginal record.
    Returns: {(parent_name, child_name): {parent_state: {child_state: prob}}}
    """
    joint_index = {}
    for j in marginal_rec.get("joints", []):
        key = (j["parent"], j["child"])
        joint_index[key] = j["table"]
    return joint_index


# ─────────────────────────────────────────────────────────────────────────────
#  Graph helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_adjacency(nodes: dict) -> dict:
    """
    Build forward adjacency: {parent: [child, ...]}
    from the nodes dict in bn_final.json.
    """
    adj = defaultdict(list)
    for child_name, info in nodes.items():
        for parent_name in info.get("parents", {}).keys():
            adj[parent_name].append(child_name)
    return dict(adj)


def build_reverse_adjacency(nodes: dict) -> dict:
    """
    Build reverse adjacency: {child: [parent, ...]}
    """
    rev = defaultdict(list)
    for child_name, info in nodes.items():
        for parent_name in info.get("parents", {}).keys():
            rev[child_name].append(parent_name)
    return dict(rev)


def get_root_nodes(nodes: dict) -> list:
    """Nodes with no parents."""
    return [n for n, info in nodes.items() if not info.get("parents")]


# ─────────────────────────────────────────────────────────────────────────────
#  Step 1 — Enumerate valid connected subgraphs
# ─────────────────────────────────────────────────────────────────────────────

def bfs_from(start: str, adj: dict, rev_adj: dict, target_size: int) -> list:
    """
    BFS traversal from start node using bidirectional edges.
    Returns list of node sets of exactly target_size nodes.
    Each set is a connected subgraph (undirected connectivity).
    """
    results   = []
    seen_sets = set()
    queue     = deque()
    queue.append(frozenset([start]))

    while queue:
        current_set = queue.popleft()

        if current_set in seen_sets:
            continue
        seen_sets.add(current_set)

        if len(current_set) == target_size:
            results.append(set(current_set))
            continue

        # expand from every node in current set using both edge directions
        for node in current_set:
            for neighbor in adj.get(node, []) + rev_adj.get(node, []):
                if neighbor not in current_set:
                    queue.append(current_set | frozenset([neighbor]))

    return results


def dfs_from(start: str, adj: dict, rev_adj: dict, target_size: int) -> list:
    """
    DFS traversal from start node using bidirectional edges.
    Returns list of node sets of exactly target_size nodes.
    """
    results   = []
    seen_sets = set()

    def dfs(current_set: frozenset):
        if current_set in seen_sets:
            return
        seen_sets.add(current_set)

        if len(current_set) == target_size:
            results.append(set(current_set))
            return

        # expand from every node in current set using both edge directions
        for node in current_set:
            for neighbor in adj.get(node, []) + rev_adj.get(node, []):
                if neighbor not in current_set:
                    dfs(current_set | frozenset([neighbor]))

    dfs(frozenset([start]))
    return results


def enumerate_subgraphs(nodes: dict, adj: dict, rev_adj: dict,
                         min_size: int, max_size: int,
                         traversal: str = "bfs") -> list:
    """
    Enumerate all valid connected subgraphs with sizes in [min_size, max_size]
    starting from every possible node.

    traversal: "bfs" | "dfs" | "both"
      "both" runs BFS and DFS from each start node and deduplicates.
    Returns list of node sets.
    """
    all_subgraphs = []
    seen          = set()

    traverse_fns = []
    if traversal in ("bfs", "both"):
        traverse_fns.append(bfs_from)
    if traversal in ("dfs", "both"):
        traverse_fns.append(dfs_from)

    for size in range(min_size, max_size + 1):
        for start_node in nodes.keys():
            for traverse_fn in traverse_fns:
                subgraphs = traverse_fn(start_node, adj, rev_adj, size)
                for sg in subgraphs:
                    key = frozenset(sg)
                    if key not in seen:
                        seen.add(key)
                        all_subgraphs.append(sg)

    return all_subgraphs


def bfs_with_required_nodes(required_nodes: set, adj: dict, rev_adj: dict,
                            target_size: int) -> list:
    """
    BFS traversal that MUST include all required_nodes.
    Expands from required_nodes to reach target_size.
    Returns list of node sets of exactly target_size that contain all required nodes.
    """
    if target_size < len(required_nodes):
        return []

    results   = []
    seen_sets = set()
    queue     = deque()
    queue.append(frozenset(required_nodes))

    while queue:
        current_set = queue.popleft()

        if current_set in seen_sets:
            continue
        seen_sets.add(current_set)

        if len(current_set) == target_size:
            results.append(set(current_set))
            continue

        # expand from every node using both edge directions
        for node in current_set:
            for neighbor in adj.get(node, []) + rev_adj.get(node, []):
                if neighbor not in current_set:
                    new_set = current_set | frozenset([neighbor])
                    if new_set not in seen_sets:
                        queue.append(new_set)

    return results


def enumerate_subgraphs_with_required(nodes: dict, adj: dict, rev_adj: dict,
                                       required_node_ids: set,
                                       min_size: int, max_size: int,
                                       traversal: str = "bfs") -> list:
    """
    Enumerate connected subgraphs that MUST include all required_node_ids.
    Only works with target sizes >= len(required_node_ids).

    Returns list of node sets containing all required nodes.
    """
    if not required_node_ids.issubset(nodes.keys()):
        missing = required_node_ids - set(nodes.keys())
        raise ValueError(f"Required nodes not in graph: {missing}")

    if min_size < len(required_node_ids):
        min_size = len(required_node_ids)

    all_subgraphs = []
    seen          = set()

    for size in range(min_size, max_size + 1):
        if traversal in ("bfs", "both"):
            subgraphs = bfs_with_required_nodes(required_node_ids, adj, rev_adj, size)
            for sg in subgraphs:
                key = frozenset(sg)
                if key not in seen:
                    seen.add(key)
                    all_subgraphs.append(sg)

        if traversal in ("dfs", "both"):
            subgraphs = dfs_with_required_nodes(required_node_ids, adj, rev_adj, size)
            for sg in subgraphs:
                key = frozenset(sg)
                if key not in seen:
                    seen.add(key)
                    all_subgraphs.append(sg)

    return all_subgraphs


def dfs_with_required_nodes(required_nodes: set, adj: dict, rev_adj: dict,
                            target_size: int) -> list:
    """
    DFS traversal that MUST include all required_nodes.
    Expands from required_nodes to reach target_size.
    """
    if target_size < len(required_nodes):
        return []

    results   = []
    seen_sets = set()

    def dfs(current_set: frozenset):
        if current_set in seen_sets:
            return
        seen_sets.add(current_set)

        if len(current_set) == target_size:
            results.append(set(current_set))
            return

        for node in current_set:
            for neighbor in adj.get(node, []) + rev_adj.get(node, []):
                if neighbor not in current_set:
                    dfs(current_set | frozenset([neighbor]))

    dfs(frozenset(required_nodes))
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Step 3 — Induce edges
# ─────────────────────────────────────────────────────────────────────────────

def induce_edges(node_subset: set, nodes: dict) -> dict:
    """
    Keep only nodes in subset.
    For each node, keep only parents that are also in the subset.
    Returns new nodes dict with induced edges.
    """
    induced = {}
    for name in node_subset:
        info        = nodes[name]
        orig_parents = info.get("parents", {})
        # keep only parents in subset
        surviving_parents = {
            p: v for p, v in orig_parents.items()
            if p in node_subset
        }
        induced[name] = {
            "states":  info["states"],
            "level":   info["level"],
            "parents": surviving_parents,
        }
        # carry prior if root
        if "prior" in info and not surviving_parents:
            induced[name]["prior"] = info["prior"]

    return induced


# ─────────────────────────────────────────────────────────────────────────────
#  Step 4 — Re-root cut nodes
# ─────────────────────────────────────────────────────────────────────────────

def reroot_cut_nodes(induced_nodes: dict,
                     original_nodes: dict,
                     marginals: dict) -> dict:
    """
    Nodes that had parents in original BN but lost ALL of them
    in the induced subgraph become new roots.
    Their prior = marginal distribution from bn_marginal_cpds.jsonl.
    """
    for name, info in induced_nodes.items():
        had_parents      = bool(original_nodes[name].get("parents"))
        has_parents_now  = bool(info.get("parents"))

        if had_parents and not has_parents_now:
            # became a new root — assign prior from marginals
            states   = info["states"]
            marginal = marginals.get(name, {})

            if marginal:
                prior = [
                    max(FLOOR, float(marginal.get(s, FLOOR)))
                    for s in states
                ]
                total = sum(prior)
                prior = [round(v / total, 6) for v in prior]
            else:
                # fallback: uniform
                u     = round(1.0 / len(states), 6)
                prior = [u] * len(states)

            info["prior"]   = prior
            info["parents"] = {}

    return induced_nodes


# ─────────────────────────────────────────────────────────────────────────────
#  Step 5 — Recompute CPDs for surviving multi-parent nodes
# ─────────────────────────────────────────────────────────────────────────────

def recover_pairwise_cpd(joint_table: dict,
                          parent_states: list,
                          child_states: list) -> list:
    """
    Recover P(child | parent) from joint using Bayes:
      P(C=c | P=p) = P(C=c, P=p) / P(P=p)
      P(P=p) = sum_c P(C=c, P=p)

    Matrix: rows = child states, cols = parent states.
    None = NaN (value < FLOOR).
    """
    matrix = []
    for cs in child_states:
        row = []
        for ps in parent_states:
            joint_val = float(joint_table.get(ps, {}).get(cs, 0.0))
            p_parent  = sum(
                float(joint_table.get(ps, {}).get(c, 0.0))
                for c in child_states
            )
            if p_parent < 1e-9:
                row.append(None)
            else:
                cond = joint_val / p_parent
                row.append(round(cond, 4) if cond >= FLOOR else None)
        matrix.append(row)
    return matrix


def recompute_cpds(induced_nodes: dict,
                   marginals: dict,
                   joint_index: dict) -> dict:
    """
    For every non-root node in the induced subgraph:
    - If it has exactly 1 surviving parent: use joint table directly
    - If it has 2+ surviving parents: recompute pairwise P(child|parent_i)
      for each surviving parent separately using joints + Bayes
    - If it lost some parents but kept others: recompute for survivors only
    """
    for name, info in induced_nodes.items():
        surviving_parents = info.get("parents", {})
        if not surviving_parents:
            continue  # root node — skip

        states = info["states"]
        new_parents = {}

        for parent_name, _ in surviving_parents.items():
            parent_states = induced_nodes[parent_name]["states"]
            key           = (parent_name, name)
            joint_table   = joint_index.get(key, {})

            if joint_table:
                matrix = recover_pairwise_cpd(
                    joint_table, parent_states, states
                )
            else:
                # no joint available — use marginals to approximate
                marginal = marginals.get(name, {})
                if marginal:
                    col   = [
                        max(FLOOR, float(marginal.get(s, FLOOR)))
                        for s in states
                    ]
                    total = sum(col)
                    col   = [round(v / total, 6) for v in col]
                else:
                    u   = round(1.0 / len(states), 6)
                    col = [u] * len(states)
                # same column for every parent state
                matrix = [[col[r] for _ in parent_states]
                          for r in range(len(states))]

            new_parents[parent_name] = {
                "parent_states": parent_states,
                "cpd_matrix":    matrix,
            }

        info["parents"] = new_parents

    return induced_nodes


# ─────────────────────────────────────────────────────────────────────────────
#  Recompute levels for subgraph
# ─────────────────────────────────────────────────────────────────────────────

def recompute_levels(induced_nodes: dict) -> dict:
    """Recompute levels from scratch for the subgraph."""
    level = {}

    def get_level(n):
        if n in level:
            return level[n]
        parents = list(induced_nodes[n].get("parents", {}).keys())
        if not parents:
            level[n] = 0
        else:
            level[n] = max(get_level(p) for p in parents) + 1
        return level[n]

    for name in induced_nodes:
        get_level(name)

    for name in induced_nodes:
        induced_nodes[name]["level"] = level[name]

    return induced_nodes


# ─────────────────────────────────────────────────────────────────────────────
#  Build one subgraph record
# ─────────────────────────────────────────────────────────────────────────────

def build_subgraph(node_subset: set,
                   parent_bn: dict,
                   parent_id: str,
                   sub_idx: int,
                   marginals: dict,
                   joint_index: dict,
                   original_nodes: dict) -> dict:
    """
    Build a complete subgraph record from a node subset.
    """
    sub_id = f"{parent_id}_sub_{sub_idx:03d}"

    # Step 3 — induce edges
    induced = induce_edges(node_subset, original_nodes)

    # Step 4 — re-root cut nodes
    induced = reroot_cut_nodes(induced, original_nodes, marginals)

    # Step 5 — recompute CPDs
    induced = recompute_cpds(induced, marginals, joint_index)

    # recompute levels
    induced = recompute_levels(induced)

    return {
        "id":             sub_id,
        "parent_id":      parent_id,
        "source_id":      parent_bn.get("source_id", ""),
        "title":          parent_bn.get("title", ""),
        "domain":         parent_bn.get("domain", ""),
        "actual_text":    parent_bn.get("actual_text", ""),
        "generated_text": parent_bn.get("generated_text", ""),
        "nodes":          induced,
        "n_nodes":        len(induced),
        "n_edges":        sum(
            len(info.get("parents", {}))
            for info in induced.values()
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Generate connected subgraphs from bn_final_updated.json"
    )
    ap.add_argument("--final",        default="bn_final_updated.json",
                    help="bn_final_updated.json (default: bn_final_updated.json)")
    ap.add_argument("--marginal",     default="bn_marginal_cpds_updated.jsonl",
                    help="bn_marginal_cpds_updated.jsonl for marginals + joints")
    ap.add_argument("--input",        default=None,
                    help="Existing subgraph JSON to merge with (preserves existing subgraphs)")
    ap.add_argument("--output",       default="bn_subgraphs_max.json")
    ap.add_argument("--target-size",  type=int, default=None,
                    help="Fixed node count per subgraph — sets both min and max size")
    ap.add_argument("--min-size",     type=int, default=3,
                    help="Minimum subgraph size (default: 5, or --target-size)")
    ap.add_argument("--max-size",     type=int, default=9,
                    help="Maximum subgraph size (default: min-size)")
    ap.add_argument("--n-subgraphs",    type=int, default=None,
                    help="Number of subgraphs to sample per BN; omit to keep all")
    ap.add_argument("--all",            dest="n_subgraphs", action="store_const",
                    const=None,
                    help="Keep all enumerated subgraphs (default when --n-subgraphs omitted)")
    ap.add_argument("--max-subgraphs",  type=int, default=5000,
                    help="Global cap on total subgraphs across all BNs (sampled proportionally)")
    ap.add_argument("--traversal",    default="bfs",
                    choices=["bfs", "dfs", "both"],
                    help="Traversal method: bfs, dfs, or both (default: bfs)")
    ap.add_argument("--required-nodes", type=int, nargs="+", default=None,
                    help="Required node IDs to include in ALL subgraphs (e.g., --required-nodes 17 32 33 45 49)")
    ap.add_argument("--only-required", action="store_true",
                    help="Create subgraph with ONLY the required nodes (no expansion)")
    ap.add_argument("--required-bns", type=str, nargs="+", default=None,
                    help="Required Bayesian Network IDs to ensure are included (e.g., --required-bns bn_0017 bn_0032 bn_0033)")
    ap.add_argument("--only-bns", type=str, nargs="+", default=None,
                    help="Generate subgraphs ONLY for these BN IDs, skip all others (e.g., --only-bns bn_0017 bn_0032 bn_0033)")
    ap.add_argument("--count-only",   action="store_true",
                    help="Only count possible subgraphs, don't generate")
    ap.add_argument("--seed",         type=int, default=42,
                    help="Random seed for reproducibility (default: 42)")
    args = ap.parse_args()

    # resolve size range
    if args.target_size is not None:
        min_size = max_size = args.target_size
    else:
        min_size = args.min_size if args.min_size is not None else 3
        max_size = args.max_size if args.max_size is not None else 8
    if min_size > max_size:
        print(f"ERROR: --min-size ({min_size}) > --max-size ({max_size})")
        return

    random.seed(args.seed)

    use_all = (args.n_subgraphs is None)  # --all mode: keep every enumerated subgraph

    # ── Load files ────────────────────────────────────────────────────────────
    for path in [args.final, args.marginal]:
        if not os.path.exists(path):
            print(f"ERROR: file not found: {path}")
            return

    if args.input and not os.path.exists(args.input):
        print(f"ERROR: input file not found: {args.input}")
        return

    data     = json.loads(open(args.final).read())
    bns      = data["bayesian_networks"]
    marginal_records = load_marginal_records(args.marginal)

    # Load existing subgraphs if input file provided
    existing_subgraphs = {}
    if args.input:
        existing_data = json.loads(open(args.input).read())
        existing_subgraphs = existing_data.get("bayesian_networks", {})
        print(f"Loaded existing subgraphs from {args.input}")
        print(f"  Existing BNs with subgraphs: {len(existing_subgraphs)}")

    size_label = (f"{min_size}" if min_size == max_size
                  else f"{min_size}–{max_size}")
    print(f"Subgraph Generator")
    print(f"  Input:       {args.final}  ({len(bns)} BNs)")
    print(f"  Marginal:    {args.marginal}")
    print(f"  Size range:  {size_label} nodes")
    print(f"  Traversal:   {args.traversal.upper()}")
    print(f"  Mode:        {'ALL subgraphs' if use_all else args.n_subgraphs}")
    if args.max_subgraphs:
        print(f"  Global cap:  {args.max_subgraphs}")
    print("-" * 55)

    # ── Enumerate (and optionally count only) ─────────────────────────────────
    required_set = set(args.required_nodes) if args.required_nodes else None
    required_bns = set(args.required_bns) if args.required_bns else None
    only_bns = set(args.only_bns) if args.only_bns else None

    if required_set:
        if args.only_required:
            print(f"\nEnumerating subgraphs with ONLY required nodes: {sorted(required_set)}...")
        else:
            print(f"\nEnumerating subgraphs (size={size_label}) with required nodes: {sorted(required_set)}...")
    if only_bns:
        print(f"\nEnumerating subgraphs ONLY for BNs: {sorted(only_bns)}")
    elif required_bns:
        print(f"Required BNs (will not be skipped): {sorted(required_bns)}")
    if not required_set and not required_bns and not only_bns:
        print(f"\nEnumerating subgraphs (size={size_label})...")

    possible_subgraphs = {}  # bn_id -> list of node sets

    for bn_id, bn in bns.items():
        # Skip if only_bns is specified and this BN is not in the list
        if only_bns and bn_id not in only_bns:
            continue

        nodes = bn.get("nodes", {})
        effective_min = len(required_set) if required_set else min_size
        is_required_bn = required_bns and bn_id in required_bns

        if len(nodes) < effective_min:
            if is_required_bn:
                print(f"  {bn_id}: only {len(nodes)} nodes — "
                      f"smaller than min-size {effective_min}, but REQUIRED BN so continuing")
                # For required BNs, relax the min_size constraint
                effective_min = min(effective_min, len(nodes))
            else:
                print(f"  {bn_id}: only {len(nodes)} nodes — "
                      f"smaller than min-size {effective_min}, skipping")
                possible_subgraphs[bn_id] = []
                continue

        adj       = build_adjacency(nodes)
        rev_adj   = build_reverse_adjacency(nodes)

        if required_set:
            if not required_set.issubset(set(nodes.keys())):
                missing = required_set - set(nodes.keys())
                if is_required_bn:
                    print(f"  {bn_id}: required nodes {missing} not in graph, but REQUIRED BN — cannot proceed")
                else:
                    print(f"  {bn_id}: required nodes {missing} not in graph, skipping")
                possible_subgraphs[bn_id] = []
                continue

            if args.only_required:
                # Create single subgraph with only required nodes (no expansion)
                subgraphs = [required_set]
            else:
                subgraphs = enumerate_subgraphs_with_required(
                    nodes, adj, rev_adj, required_set, effective_min, max_size, args.traversal
                )
        else:
            subgraphs = enumerate_subgraphs(
                nodes, adj, rev_adj, effective_min, max_size, args.traversal
            )
        possible_subgraphs[bn_id] = subgraphs
        print(f"  {bn_id}: {len(subgraphs)} subgraphs  "
              f"({len(nodes)} nodes, {bn.get('title','')[:40]})")

    total_possible = sum(len(v) for v in possible_subgraphs.values())
    print(f"\nTotal enumerated subgraphs: {total_possible}")
    if not use_all:
        print(f"Requested per BN:           {args.n_subgraphs}")

    if args.count_only:
        print("\n(--count-only mode: stopping here)")
        return

    # ── Validate n_subgraphs ──────────────────────────────────────────────────
    if not use_all:
        for bn_id, sgs in possible_subgraphs.items():
            count = len(sgs)
            if count > 0 and args.n_subgraphs > count:
                print(f"\nERROR: {bn_id} only has {count} possible subgraphs "
                      f"but you requested {args.n_subgraphs}.")
                print(f"  Re-run with --n-subgraphs <= {count} or --all")
                return

    # ── Apply global cap (proportional sampling across BNs) ───────────────────
    if args.max_subgraphs and total_possible > args.max_subgraphs:
        # Separate required BNs from others
        required_bn_subgraphs = {}
        other_pool = []

        for bn_id, sgs in possible_subgraphs.items():
            if required_bns and bn_id in required_bns:
                # Keep all subgraphs from required BNs
                required_bn_subgraphs[bn_id] = sgs
            else:
                # Add optional BNs to sampling pool
                for sg in sgs:
                    other_pool.append((bn_id, sg))

        # Calculate budget after reserving required BNs
        required_count = sum(len(sgs) for sgs in required_bn_subgraphs.values())
        remaining_budget = max(0, args.max_subgraphs - required_count)

        if remaining_budget > 0 and other_pool:
            sampled_other = random.sample(other_pool, min(remaining_budget, len(other_pool)))
        else:
            sampled_other = []

        # Rebuild per-BN lists
        possible_subgraphs = dict(required_bn_subgraphs)
        for bn_id, sg in sampled_other:
            if bn_id not in possible_subgraphs:
                possible_subgraphs[bn_id] = []
            possible_subgraphs[bn_id].append(sg)

        final_count = sum(len(sgs) for sgs in possible_subgraphs.values())
        print(f"\nGlobal cap applied: {total_possible} → {final_count} subgraphs "
              f"(reserved {required_count} for required BNs)")

    # ── Generate subgraphs ────────────────────────────────────────────────────
    mode_label = "all" if use_all else str(args.n_subgraphs)
    print(f"\nGenerating {mode_label} subgraphs per BN...")
    all_subgraphs = {}
    n_ok = n_skip = 0

    for bn_id, bn in bns.items():
        candidates = possible_subgraphs.get(bn_id, [])
        if not candidates:
            continue

        nodes        = bn.get("nodes", {})
        source_id    = bn.get("source_id", "")
        marginal_rec = marginal_records.get(source_id, {})
        marginals    = get_marginals(marginal_rec)
        joint_index  = get_joint_index(marginal_rec)

        selected = candidates if use_all else random.sample(candidates, args.n_subgraphs)

        print(f"\n  {bn_id} ({bn.get('title','')[:40]}) — {len(selected)} subgraphs")

        for idx, node_subset in enumerate(selected, 1):
            try:
                sub = build_subgraph(
                    node_subset    = node_subset,
                    parent_bn      = bn,
                    parent_id      = bn_id,
                    sub_idx        = idx,
                    marginals      = marginals,
                    joint_index    = joint_index,
                    original_nodes = nodes,
                )
                sub_id                = sub["id"]
                all_subgraphs[sub_id] = sub
                print(f"    {sub_id}: nodes={sub['n_nodes']} "
                      f"edges={sub['n_edges']}  "
                      f"nodes: {list(node_subset)[:3]}...")
                n_ok += 1
            except Exception as e:
                print(f"    ERROR subgraph {idx}: {e}")
                n_skip += 1

    # ── Write output ──────────────────────────────────────────────────────────
    # Merge with existing subgraphs if input was provided
    final_subgraphs = all_subgraphs
    if args.input:
        final_subgraphs = dict(existing_subgraphs)
        final_subgraphs.update(all_subgraphs)
        print(f"\nMerged new subgraphs with existing ones")
        print(f"  Existing: {len(existing_subgraphs)} BNs")
        print(f"  Generated: {len(all_subgraphs)} BNs")
        print(f"  Total: {len(final_subgraphs)} BNs")

    output = {
        "metadata": {
            "total":       len(final_subgraphs),
            "source":      args.final,
            "min_size":    min_size,
            "max_size":    max_size,
            "n_subgraphs":   None if use_all else args.n_subgraphs,
            "max_subgraphs": args.max_subgraphs,
            "all_mode":      use_all,
            "traversal":     args.traversal,
            "seed":        args.seed,
            "merged_from_input": args.input,
        },
        "bayesian_networks": final_subgraphs,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\n{'='*55}")
    print(f"Done.  Generated={n_ok}  Skipped={n_skip}")
    print(f"Output: {args.output}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()