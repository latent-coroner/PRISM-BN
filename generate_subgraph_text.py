"""
generate_subgraph_texts.py
===========================
Reads bn_subgraphs_random.json, formats each subgraph into a structured
prompt, calls the Anthropic API with the provided SYSTEM_PROMPT, and
saves the generated texts back into a new JSON file.

Output schema (per subgraph):
  {
    "id":             "unique_00001",
    "parent_id":      "bn_00001",
    "domain":         "medical",
    "title":          "...",
    "actual_text":    "...",          # carried over from parent BN
    "generated_text": "...",          # NEW: generated from subgraph structure
    "nodes":          { ... },
    "n_nodes":        5,
    "n_edges":        4,
  }

Usage:
    python generate_subgraph_texts.py
    python generate_subgraph_texts.py --input  bn_subgraphs_random.json
                                      --output bn_subgraphs_with_text.json
    python generate_subgraph_texts.py --batch-size 5 --max-subgraphs 100
    python generate_subgraph_texts.py --resume   # skip already-generated entries
    python generate_subgraph_texts.py --start-from 540  # skip first 540 samples
"""

import os, sys, json, time, argparse
from collections import defaultdict

import anthropic

def validate_coverage(sg: dict, generated_text: str) -> list[str]:
    """Returns list of missing nodes/states from generated text."""
    missing = []
    text_lower = generated_text.lower()
    for node_name, info in sg["nodes"].items():
        if node_name.lower() not in text_lower:
            missing.append(f"NODE: {node_name}")
        for state in info.get("states", []):
            if state.lower() not in text_lower:
                missing.append(f"STATE: {node_name}={state}")
    return missing

# ─────────────────────────────────────────────────────────────────────────────
#  System prompt (exact copy from specification)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert at writing natural language descriptions of Bayesian Networks.

You will be given a Bayesian Network with nodes, states, causal connections, and conditional probabilities.
Your job is to write a coherent, informative paragraph describing the causal structure and probability relationships.

STRICT RULES — all must be followed:

1. COMPLETENESS — NODES AND STATES:
   Every node name AND every state of every node listed under NODES AND STATES
   must appear somewhere in your text. Do not skip any node or any state.
   Extra concepts beyond the BN are allowed — just do not miss any BN content.

2. EDGE COMPLETENESS:
   Every causal connection listed under CAUSAL CONNECTIONS must be described
   in the text. The exact same number of causal relationships must appear.
   Do not skip any edge. Do not invent edges that are not listed.

3. CPD CONSISTENCY:
   The language strength used must reflect the CPD magnitudes:
   - CPD ≥ 0.70 → use strong language: "strongly", "almost always", "predominantly"
   - CPD 0.40–0.69 → use moderate language: "often", "likely", "tends to"
   - CPD 0.10–0.39 → use weak language: "sometimes", "occasionally", "may"
   - CPD < 0.10 → use rare language: "rarely", "seldom", "in few cases"
   If you choose to write an explicit probability (e.g. "70% of the time"),
   it must exactly match the CPD value provided. Best practice is qualitative
   language to avoid precision errors — but explicit numbers are allowed if correct.

4. NATURAL LANGUAGE:
   Write in flowing prose, not bullet points. Should read like a paragraph
   from a research article or textbook. It shouldn't write in the text what the nodes are. The terms might be there but it should be in a natural way, not like "the node X has states a,b,c". It should be more like "X can take on the values a,b,c, with a being the most common".

5. LENGTH: 100–1000 words. Must cover all nodes, states, and edges. And don't add extra content beyond the BN. So you must be concise but also complete.
After generation validate that all nodes, states, and edges are mentioned in the text. If any are missing add them in new sentences at the end of the text.
Return ONLY the text. No preamble, no labels, no explanation."""


# ─────────────────────────────────────────────────────────────────────────────
#  Prompt builder
# ─────────────────────────────────────────────────────────────────────────────

def cpd_strength_label(val) -> str:
    """Map a CPD float to a qualitative strength label."""
    if val is None:
        return "unknown"
    try:
        v = float(val)
    except (TypeError, ValueError):
        return "unknown"
    if v >= 0.70:
        return "strong"
    elif v >= 0.40:
        return "moderate"
    elif v >= 0.10:
        return "weak"
    else:
        return "rare"


def format_cpd_matrix(parent_name: str, parent_states: list,
                       child_name: str, child_states: list,
                       matrix: list) -> str:
    """
    Format a CPD matrix into readable lines:
      P(child=s | parent=p) = v  [strength]
    """
    lines = []
    for ci, cs in enumerate(child_states):
        for pi, ps in enumerate(parent_states):
            try:
                val = matrix[ci][pi]
            except (IndexError, TypeError):
                val = None
            if val is None:
                continue
            label = cpd_strength_label(val)
            lines.append(
                f"  P({child_name}={cs} | {parent_name}={ps}) = {val:.4f}  [{label}]"
            )
    return "\n".join(lines)


def format_prior(node_name: str, states: list, prior: list) -> str:
    """Format root node prior as readable lines."""
    lines = []
    for s, p in zip(states, prior):
        if p is None:
            continue
        label = cpd_strength_label(p)
        lines.append(f"  P({node_name}={s}) = {p:.4f}  [{label}]")
    return "\n".join(lines)


def build_user_prompt(sg: dict) -> str:
    """
    Convert a subgraph record into a structured user prompt for the LLM.
    Sections:
      TITLE
      NODES AND STATES
      CAUSAL CONNECTIONS
      CONDITIONAL PROBABILITY TABLES
    """
    nodes  = sg.get("nodes", {})
    title  = sg.get("title", "Unknown")
    domain = sg.get("domain", "unknown")

    lines = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines.append(f"TITLE: {title}")
    lines.append(f"DOMAIN: {domain}")
    lines.append("")

    # ── Nodes and states ──────────────────────────────────────────────────────
    lines.append("NODES AND STATES:")
    for node_name, info in nodes.items():
        states = info.get("states", [])
        lines.append(f"  {node_name}: [{', '.join(states)}]")
    lines.append("")

    # ── Causal connections ────────────────────────────────────────────────────
    lines.append("CAUSAL CONNECTIONS:")
    edges_found = False
    for child_name, info in nodes.items():
        parents = info.get("parents", {})
        for parent_name in parents:
            lines.append(f"  {parent_name}  →  {child_name}")
            edges_found = True
    if not edges_found:
        lines.append("  (no edges — all nodes are roots)")
    lines.append("")

    # ── CPD tables ────────────────────────────────────────────────────────────
    lines.append("CONDITIONAL PROBABILITY TABLES:")
    for node_name, info in nodes.items():
        states  = info.get("states", [])
        parents = info.get("parents", {})

        if not parents:
            # root node — show prior
            prior = info.get("prior", [])
            if prior:
                lines.append(f"\n[{node_name}]  (root — prior distribution)")
                lines.append(format_prior(node_name, states, prior))
            else:
                lines.append(f"\n[{node_name}]  (root — no prior available)")
        else:
            # non-root — show CPD per parent
            for parent_name, p_info in parents.items():
                parent_states = p_info.get("parent_states", [])
                matrix        = p_info.get("cpd_matrix", [])
                lines.append(
                    f"\n[{node_name} | {parent_name}]"
                )
                if matrix:
                    lines.append(
                        format_cpd_matrix(
                            parent_name, parent_states,
                            node_name,   states,
                            matrix
                        )
                    )
                else:
                    lines.append("  (no CPD data available)")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  API call
# ─────────────────────────────────────────────────────────────────────────────

def generate_text(client: anthropic.Anthropic, user_prompt: str,
                  max_retries: int = 1, retry_delay: float = 5.0) -> str:
    """
    Call Claude API with retry logic.
    Returns generated text string or raises after max_retries.
    """
    for attempt in range(1, max_retries + 1):
        try:
            response = client.messages.create(
                model      = os.getenv("EXTRACTION_MODEL", "extraction-model"),
                max_tokens = 2048,
                system     = SYSTEM_PROMPT,
                messages   = [{"role": "user", "content": user_prompt}],
            )

            return response.content[0].text.strip()
        except anthropic.RateLimitError:
            if attempt < max_retries:
                wait = retry_delay * attempt
                print(f"    Rate limit hit — waiting {wait:.0f}s (attempt {attempt}/{max_retries})")
                time.sleep(wait )
            else:
                raise
        except anthropic.APIError as e:
            if attempt < max_retries:
                print(f"    API error: {e} — retrying ({attempt}/{max_retries})")
                time.sleep(retry_delay)
            else:
                raise


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Generate natural language texts for subgraphs via Claude API"
    )
    ap.add_argument("--input",         default="bn_subgraphs_maxv3.json",
                    help="Input subgraphs JSON (default: bn_subgraphs_max.json)")
    ap.add_argument("--output",        default="prism_bn.json",
                    help="Output JSON with generated texts (default: bn_subgraphs_with_text_max.json)")
    ap.add_argument("--max-subgraphs", type=int, default=None,
                    help="Max number of subgraphs to process (default: all)")
    ap.add_argument("--batch-size",    type=int, default=10,
                    help="Save progress every N subgraphs (default: 10)")
    ap.add_argument("--delay",         type=float, default=0.5,
                    help="Seconds to wait between API calls (default: 0.5)")
    ap.add_argument("--resume",        action="store_true",
                    help="Skip subgraphs that already have generated_text in output file")
    ap.add_argument("--start-from",    type=int, default=0,
                    help="Skip the first N subgraphs and start processing from index N (0-based, default: 0)")
    ap.add_argument("--start-from-id", type=str, default=None,
                    help="Start processing from a specific subgraph ID (e.g., bn_00028_sub_016)")
    ap.add_argument("--missing-only",  action="store_true",
                    help="Only process IDs from input that are NOT in the output file (useful for incremental updates)")
    args = ap.parse_args()

    # ── Load input ────────────────────────────────────────────────────────────
    if not os.path.exists(args.input):
        print(f"ERROR: {args.input} not found."); sys.exit(1)

    print(f"Loading {args.input} ...")
    data      = json.loads(open(args.input, encoding="utf-8").read())
    subgraphs = data.get("bayesian_networks", {})
    metadata  = data.get("metadata", {})
    print(f"  {len(subgraphs)} subgraphs loaded.")

    # ── Resume / Missing-only: load existing output if present ────────────────
    already_done = set()
    results      = {}  # only store processed subgraphs
    existing_metadata = metadata

    if args.resume and os.path.exists(args.output):
        print(f"  Resuming from {args.output} ...")
        existing = json.loads(open(args.output, encoding="utf-8").read())
        for sg_id, sg in existing.get("bayesian_networks", {}).items():
            if sg.get("generated_text", "").strip():
                already_done.add(sg_id)
                results[sg_id] = sg

        print(f"  {len(already_done)} subgraphs already have generated text — skipping.")

    elif args.missing_only and os.path.exists(args.output):
        print(f"  Loading existing output from {args.output} ...")
        existing = json.loads(open(args.output, encoding="utf-8").read())
        existing_bns = existing.get("bayesian_networks", {})
        existing_metadata = existing.get("metadata", metadata)

        # Keep all existing entries (already processed) in results
        for sg_id, sg in existing_bns.items():
            results[sg_id] = sg
            already_done.add(sg_id)

        print(f"  {len(already_done)} subgraphs already in output file.")

    # ── Select subgraphs to process ───────────────────────────────────────────
    to_process = [
        (sg_id, sg)
        for sg_id, sg in subgraphs.items()
        if sg_id not in already_done
    ]
    if args.start_from_id:
        # Find the index of the specified ID and start from there
        start_idx = None
        for idx, (sg_id, _) in enumerate(to_process):
            if sg_id == args.start_from_id:
                start_idx = idx
                break
        if start_idx is not None:
            to_process = to_process[start_idx:]
            print(f"  Starting from ID: {args.start_from_id} (index {start_idx})")
        else:
            print(f"  WARNING: ID {args.start_from_id} not found in to-process list")
    elif args.start_from > 0:
        to_process = to_process[args.start_from:]
    if args.max_subgraphs is not None:
        to_process = to_process[:args.max_subgraphs]

    total = len(to_process)
    print(f"\n  To process : {total} subgraphs")
    print(f"  Batch save : every {args.batch_size}")
    print(f"  Delay      : {args.delay}s between calls")

    if total == 0:
        print("Nothing to do."); sys.exit(0)

    # ── Init Anthropic client ─────────────────────────────────────────────────
    client = anthropic.Anthropic(api_key =os.getenv("ANTHROPIC_API_KEY"))   # reads ANTHROPIC_API_KEY from env

    # ── Generation loop ───────────────────────────────────────────────────────
    n_ok = n_fail = 0

    def save_checkpoint():
        out = {
            "metadata": {
                **existing_metadata,
                "generated_total": sum(1 for sg in results.values() if sg.get("generated_text", "").strip()),
                "generation_source": args.input,
            },
            "bayesian_networks": results,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\n{'─'*60}")
    print(f"  {'#':>5}  {'ID':<18}  {'nodes':>5}  {'edges':>5}  status")
    print(f"{'─'*60}")

    for i, (sg_id, sg) in enumerate(to_process, 1):
        n_nodes = sg.get("n_nodes", 0)
        n_edges = sg.get("n_edges", 0)

        results[sg_id] = sg

        try:
            user_prompt    = build_user_prompt(sg)
            generated_text = generate_text(client, user_prompt)
            missing = validate_coverage(sg, generated_text)
            if missing:
                print(f"  WARNING: Missing coverage: {missing}")
            results[sg_id]["generated_text"] = generated_text
            n_ok += 1
            status = f"OK  ({len(generated_text.split())} words)"

        except Exception as e:
            results[sg_id]["generated_text"] = ""
            n_fail += 1
            status = f"FAIL: {str(e)[:40]}"

        print(f"  {i:>5}  {sg_id:<18}  {n_nodes:>5}  {n_edges:>5}  {status}")

        # checkpoint save
        if i % args.batch_size == 0:
            save_checkpoint()
            print(f"  [checkpoint saved at {i}/{total}]")

        if args.delay > 0 and i < total:
            time.sleep(args.delay)

    # ── Final save ────────────────────────────────────────────────────────────
    save_checkpoint()
    size_mb = os.path.getsize(args.output) / (1024 * 1024)

    print(f"\n{'='*60}")
    print(f"  Done.  OK={n_ok}  Failed={n_fail}")
    print(f"  Output → {args.output}  ({size_mb:.2f} MB)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()