"""
bn_extractor_marginal.py
========================
Alternate BN extractor. Instead of asking Claude for raw conditional scores
(Pass 4 in the original), Claude provides:

  Pass 3 — Marginal probabilities P(node_state) for ALL nodes
  Pass 4 — Joint probabilities P(node_state, parent_state) for each edge

Then cpd_calculator_marginal.py recovers:
  P(child | parent) = P(child, parent) / P(parent)

This is the "marginal-first" approach — Claude reasons about how common
each state is in the world, rather than estimating conditional strengths.

Pipeline:
  Pass 1 — nodes + states          (same as original)
  Pass 2 — edges + DAG validation  (same as original)
  Pass 3 — marginals for ALL nodes (different — not just roots)
  Pass 4 — joint probabilities      (different — replaces raw scores)

Output: bn_marginal_raw.jsonl

Usage:
    python bn_extractor_marginal.py
    python bn_extractor_marginal.py --output bn_marginal_raw.jsonl --per-domain 10
"""

import os
import re
import json
import time
import random
import argparse
from collections import defaultdict

import anthropic
import networkx as nx
import wikipedia
import requests as req_lib


# ─────────────────────────────────────────────────────────────────────────────
#  Pass 1 — Entity Extraction
# ─────────────────────────────────────────────────────────────────────────────

PASS1_SYSTEM = """You are an expert at identifying causal concepts in text.
Your job: extract causal nodes and their possible states from an article.

Rules:
- Extract AS MANY nodes as the text warrants — anywhere from 1 to 20 nodes.
- Group same-type concepts into ONE node with multiple states.
  e.g. "recession" and "inflation" are both states of node "economic shock".
- Always add "none" as the last state of every node (represents absence/baseline).
- Node names must be abstract category labels (e.g. "economic shock", not "recession").
- State names must be concrete specific values (e.g. "recession", "inflation").
- 2–6 concrete states per node plus "none".
- Return ONLY valid JSON, no markdown, no explanation."""


def pass1_prompt(text: str) -> str:
    return f"""Extract causal nodes and states from this article.
Extract as many nodes as the text supports (1–20).

Article:
\"\"\"
{text[:4000]}
\"\"\"

Return ONLY this JSON structure:
{{
  "nodes": [
    {{"name": "economic shock",  "states": ["recession", "inflation", "stagflation", "none"]}},
    {{"name": "social outcome",  "states": ["unemployment", "poverty", "none"]}}
  ]
}}

Include every causally relevant concept in the article."""


# ─────────────────────────────────────────────────────────────────────────────
#  Pass 2 — Causal Structure
# ─────────────────────────────────────────────────────────────────────────────

PASS2_SYSTEM = """You are an expert at identifying causal relationships between concepts.
Your job: decide which nodes causally influence which other nodes.

Rules:
- Only add an edge A → B if A directly causes or influences B.
- Do NOT add edges just because two concepts co-occur.
- The result MUST be a Directed Acyclic Graph (no cycles).
- Return ONLY valid JSON, no markdown, no explanation."""


def pass2_prompt(nodes: list, text: str) -> str:
    node_names = [n["name"] for n in nodes]
    return f"""Given these nodes extracted from an article, identify causal edges.

Nodes: {json.dumps(node_names)}

Article (for context):
\"\"\"
{text[:2000]}
\"\"\"

Return ONLY this JSON structure:
{{
  "edges": [
    {{"parent": "economic shock", "child": "social outcome"}},
    {{"parent": "economic shock", "child": "policy response"}}
  ]
}}

Use only node names from the list above. No cycles allowed."""


# ─────────────────────────────────────────────────────────────────────────────
#  Pass 3 — Marginal Probabilities for ALL nodes
# ─────────────────────────────────────────────────────────────────────────────

PASS3_SYSTEM = """You are an expert at estimating real-world base rates and frequencies.
Your job: assign marginal probability distributions to ALL nodes in the network.

A marginal probability P(node_state) means:
  "In the world described by this article, how common is this state overall?"
  — NOT conditioned on any other node.
  — Think of it as: if you randomly sampled a scenario from this domain,
    how often would you encounter each state?

Rules:
- Use BOTH the article text AND your general world knowledge.
- Each distribution must sum to exactly 1.0.
- "none" state = baseline/absence. Give it realistic weight.
- Be realistic — base rates should reflect real-world frequencies.
- For child nodes, still give the UNCONDITIONAL marginal (ignore parents here).
- Return ONLY valid JSON, no markdown, no explanation."""


def pass3_prompt(nodes: list, text: str) -> str:
    return f"""Assign marginal probability distributions to ALL nodes.

All nodes:
{json.dumps(nodes, indent=2)}

Article (for context):
\"\"\"
{text[:2000]}
\"\"\"

Return ONLY this JSON structure:
{{
  "marginals": [
    {{
      "node": "economic shock",
      "distribution": {{"recession": 0.35, "inflation": 0.40, "none": 0.25}},
      "rationale": "historically recessions and inflation roughly equally common"
    }},
    {{
      "node": "social outcome",
      "distribution": {{"unemployment": 0.30, "poverty": 0.25, "none": 0.45}},
      "rationale": "unemployment affects ~30% of populations during downturns"
    }}
  ]
}}

Every state of every node must have a probability. All distributions must sum to 1.0.
These are UNCONDITIONAL marginals — do not condition on any parent node."""


# ─────────────────────────────────────────────────────────────────────────────
#  Pass 4 — Joint Probabilities
# ─────────────────────────────────────────────────────────────────────────────

PASS4_SYSTEM = """You are an expert at estimating how often two events co-occur.
Your job: for each causal edge A → B, estimate the joint probability
P(A = a, B = b) for every combination of states.

A joint probability P(A=a, B=b) means:
  "How often do BOTH of these states occur together in the same scenario?"

Rules:
- These are JOINT probabilities — P(A=a AND B=b) simultaneously.
- The full joint table for an edge must sum to 1.0 across ALL combinations.
  e.g. for edge A→B where A has 2 states and B has 3 states:
  sum of all 2×3 = 6 joint values must equal 1.0
- Use the article text as primary evidence; world knowledge as secondary.
- "none" states can co-occur with other states (e.g. no economic shock but still poverty).
- Return ONLY valid JSON, no markdown, no explanation."""


def pass4_prompt(nodes: list, edges: list, text: str) -> str:
    node_map = {n["name"]: n["states"] for n in nodes}
    edge_requests = []

    for edge in edges:
        parent   = edge["parent"]
        child    = edge["child"]
        p_states = node_map.get(parent, [])
        c_states = node_map.get(child,  [])
        if not p_states or not c_states:
            continue

        combos = []
        for ps in p_states:
            for cs in c_states:
                combos.append({"parent_state": ps, "child_state": cs})

        edge_requests.append({
            "parent":       parent,
            "child":        child,
            "combinations": combos,
            "note":         f"all {len(combos)} values must sum to 1.0"
        })

    return f"""Assign joint probabilities P(parent_state, child_state) for each edge.

Edges to score ({len(edge_requests)} edges):
{json.dumps(edge_requests, indent=2)}

Article (for context):
\"\"\"
{text[:2000]}
\"\"\"

Return ONLY this JSON structure:
{{
  "joints": [
    {{
      "parent": "economic shock",
      "child":  "social outcome",
      "table": {{
        "recession":  {{"unemployment": 0.18, "poverty": 0.12, "none": 0.05}},
        "inflation":  {{"unemployment": 0.09, "poverty": 0.08, "none": 0.23}},
        "none":       {{"unemployment": 0.03, "poverty": 0.05, "none": 0.17}}
      }}
    }}
  ]
}}

CRITICAL: For each edge, ALL values in the table must sum to exactly 1.0.
The sum is across ALL parent_state × child_state combinations together."""


# ─────────────────────────────────────────────────────────────────────────────
#  Pass 5 — Full Multi-Parent Joints  (exact CPDs for 2-3 parent nodes)
# ─────────────────────────────────────────────────────────────────────────────

PASS5_SYSTEM = """You are an expert at estimating how often multiple events co-occur simultaneously.
Your job: for each child node that has multiple parents, estimate the FULL joint probability
P(child, parent_1, parent_2, ...) for every combination of states.

Rules:
- These are JOINT probabilities — all variables occurring simultaneously.
- The ENTIRE table for each child must sum to exactly 1.0 across ALL leaf values.
- Nest by parent states using keys of the form "node_name=state_name".
- Use the article text as primary evidence; world knowledge as secondary.
- Return ONLY valid JSON, no markdown, no explanation."""


def pass5_prompt(nodes: list, edges: list, text: str,
                 target_children: list = None) -> str:
    node_map   = {n["name"]: n["states"] for n in nodes}
    parent_map = defaultdict(list)
    for e in edges:
        parent_map[e["child"]].append(e["parent"])

    requests = []
    for child_name, parent_names in parent_map.items():
        if len(parent_names) < 2 or len(parent_names) > 3:
            continue
        if target_children is not None and child_name not in target_children:
            continue
        child_states = node_map.get(child_name, [])
        parents_info = [{"name": pn, "states": node_map.get(pn, [])}
                        for pn in parent_names]
        n_combos = len(child_states)
        for pi in parents_info:
            n_combos *= len(pi["states"])
        requests.append({
            "child":        child_name,
            "child_states": child_states,
            "parents":      parents_info,
            "note":         f"all {n_combos} leaf values must sum to 1.0",
        })

    if not requests:
        return ""

    return f"""Assign FULL joint probabilities for child nodes with multiple parents.

For each child, fill in P(child=c, parent_1=p1, parent_2=p2) for EVERY combination.
Nest the table: parent_1_state → parent_2_state → {{child_state: prob}}.

Nodes to score ({len(requests)} nodes):
{json.dumps(requests, indent=2)}

Article (for context):
\"\"\"
{text[:2000]}
\"\"\"

Return ONLY this JSON structure:
{{
  "joints": [
    {{
      "child": "long-term impact",
      "table": {{
        "economic shock=recession": {{
          "policy response=austerity": {{"stagnation": 0.08, "recovery": 0.02, "none": 0.01}},
          "policy response=stimulus":  {{"stagnation": 0.03, "recovery": 0.07, "none": 0.01}},
          "policy response=none":      {{"stagnation": 0.05, "recovery": 0.03, "none": 0.01}}
        }},
        "economic shock=inflation": {{
          "policy response=austerity": {{"stagnation": 0.06, "recovery": 0.03, "none": 0.02}},
          "policy response=stimulus":  {{"stagnation": 0.02, "recovery": 0.08, "none": 0.02}},
          "policy response=none":      {{"stagnation": 0.04, "recovery": 0.04, "none": 0.02}}
        }},
        "economic shock=none": {{
          "policy response=austerity": {{"stagnation": 0.03, "recovery": 0.02, "none": 0.04}},
          "policy response=stimulus":  {{"stagnation": 0.01, "recovery": 0.05, "none": 0.04}},
          "policy response=none":      {{"stagnation": 0.02, "recovery": 0.03, "none": 0.08}}
        }}
      }}
    }}
  ]
}}

CRITICAL: For each child, ALL leaf values across its entire table must sum to exactly 1.0."""


# ─────────────────────────────────────────────────────────────────────────────
#  Claude caller
# ─────────────────────────────────────────────────────────────────────────────

def call_claude(client, system, user, model,
                max_tokens=2000, max_retries=3) -> dict | None:
    for attempt in range(max_retries):
        try:
            msg = client.messages.create(
                model      = model,
                max_tokens = max_tokens,
                system     = system,
                messages   = [{"role": "user", "content": user}],
            )
            if msg.stop_reason == "max_tokens":
                print(f"      [max_tokens truncation on attempt {attempt+1} — response cut off, skipping]")
                time.sleep(1)
                continue
            text_block = next((b for b in msg.content if b.type == "text"), None)
            if text_block is None:
                print(f"      [No text block on attempt {attempt+1}; stop_reason={msg.stop_reason}]")
                time.sleep(1)
                continue
            raw = text_block.text.strip()
            # Extract JSON from a fenced code block if present anywhere in the response
            fence_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
            if fence_match:
                raw = fence_match.group(1).strip()
            else:
                # Strip leading/trailing fences (no-preamble path)
                raw = re.sub(r'^```[a-zA-Z]*\n?', '', raw)
                raw = re.sub(r'\n?```$', '', raw).strip()
                # Fall back: slice from first { to last }
                start, end = raw.find('{'), raw.rfind('}')
                if start != -1 and end != -1:
                    raw = raw[start:end + 1]
            if not raw:
                print(f"      [Empty response body on attempt {attempt+1}]")
                time.sleep(1)
                continue
            return json.loads(raw)

        except json.JSONDecodeError as e:
            print(f"      [JSON error attempt {attempt+1}]: {e}")
            print(f"      [Raw ({len(raw)} chars)]: {repr(raw[:300])}")
            time.sleep(1)
        except anthropic.RateLimitError:
            print(f"      [Rate limit — waiting 60s]")
            time.sleep(60)
        except Exception as e:
            print(f"      [API error attempt {attempt+1}]: {e}")
            time.sleep(2)
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  DAG validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_dag(nodes, edges):
    node_names = {n["name"] for n in nodes}
    G = nx.DiGraph()
    G.add_nodes_from(node_names)
    valid_edges = []
    for edge in edges:
        p, c = edge.get("parent"), edge.get("child")
        if p not in node_names or c not in node_names:
            continue
        G.add_edge(p, c)
        if nx.is_directed_acyclic_graph(G):
            valid_edges.append(edge)
        else:
            print(f"      [DAG] Cycle removed: {p} → {c}")
            G.remove_edge(p, c)
    topo = list(nx.topological_sort(G)) if nx.is_directed_acyclic_graph(G) else []
    return nx.is_directed_acyclic_graph(G), valid_edges, topo


# ─────────────────────────────────────────────────────────────────────────────
#  Validators
# ─────────────────────────────────────────────────────────────────────────────

def validate_marginals(marginals: list, nodes: list) -> dict:
    """
    Returns a dict: node_name → {state: probability}.
    Normalizes rows that don't sum to 1.0.
    """
    node_map  = {n["name"]: n["states"] for n in nodes}
    validated = {}

    for m in marginals:
        name = m.get("node")
        dist = m.get("distribution", {})
        if not name or not dist:
            continue

        states = node_map.get(name, list(dist.keys()))
        filled = {s: max(0.001, float(dist.get(s, 0.001))) for s in states}
        total  = sum(filled.values())
        if abs(total - 1.0) > 0.02:
            print(f"      [Marginal] Normalizing {name}: sum={total:.3f}")
            filled = {s: round(v / total, 6) for s, v in filled.items()}

        validated[name] = filled

    return validated


def validate_joints(joints: list, nodes: list, edges: list) -> list:
    """
    Validates and normalizes joint tables.
    Returns list of dicts: {parent, child, table}
    where table[parent_state][child_state] = probability.
    """
    node_map  = {n["name"]: n["states"] for n in nodes}
    validated = []

    for j in joints:
        parent = j.get("parent")
        child  = j.get("child")
        table  = j.get("table", {})
        if not parent or not child or not table:
            continue

        p_states = node_map.get(parent, [])
        c_states = node_map.get(child,  [])

        # normalize entire table to sum to 1.0
        all_vals = []
        for ps in p_states:
            row = table.get(ps, {})
            for cs in c_states:
                all_vals.append(max(0.0001, float(row.get(cs, 0.0001))))

        total = sum(all_vals)
        if total == 0:
            continue

        # rebuild normalized table
        norm_table = {}
        for ps in p_states:
            norm_table[ps] = {}
            row = table.get(ps, {})
            for cs in c_states:
                raw_val = max(0.0001, float(row.get(cs, 0.0001)))
                norm_table[ps][cs] = round(raw_val / total, 6)

        if abs(sum(v for row in norm_table.values()
                   for v in row.values()) - 1.0) > 0.01:
            print(f"      [Joint] Normalization issue: {parent}→{child}")

        validated.append({
            "parent": parent,
            "child":  child,
            "table":  norm_table,
        })

    return validated


# ─────────────────────────────────────────────────────────────────────────────
#  Pass 4 chunking for large BNs
# ─────────────────────────────────────────────────────────────────────────────

def pass4_tokens_needed(n_edges: int, avg_states: int = 4) -> int:
    # each joint table entry ~40 tokens, avg n_states^2 combos per edge
    return max(2000, min(8000, n_edges * avg_states * avg_states * 40))


def pass4_chunked(nodes, edges, text, client, model, chunk_size=10) -> list:
    all_joints = []
    for i in range(0, len(edges), chunk_size):
        chunk   = edges[i: i + chunk_size]
        max_tok = pass4_tokens_needed(len(chunk))
        result  = call_claude(client, PASS4_SYSTEM,
                              pass4_prompt(nodes, chunk, text),
                              model, max_tokens=max_tok)
        if result and "joints" in result:
            all_joints.extend(result["joints"])
        else:
            print(f"      [Pass 4] chunk {i//chunk_size+1} failed")
    return all_joints


# ─────────────────────────────────────────────────────────────────────────────
#  4-pass extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_bn_marginal(text: str, client, model: str,
                        chunk_size: int = 10) -> dict | None:

    # Pass 1 — nodes + states
    p1_tokens = min(4000, max(1500, len(text.split()) // 5))
    p1 = call_claude(client, PASS1_SYSTEM, pass1_prompt(text),
                     model, max_tokens=p1_tokens)
    if not p1 or not p1.get("nodes"):
        print("      [Pass 1] FAILED")
        return None
    nodes = p1["nodes"]
    print(f"      Pass 1 OK — {len(nodes)} nodes")

    # Pass 2 — edges + DAG validation
    p2_tokens = min(4000, max(1500, len(nodes) * 80))
    p2 = call_claude(client, PASS2_SYSTEM, pass2_prompt(nodes, text),
                     model, max_tokens=p2_tokens)
    if not p2 or not p2.get("edges"):
        print("      [Pass 2] FAILED")
        return None
    _, edges, topo = validate_dag(nodes, p2["edges"])
    if not edges:
        print("      [Pass 2] FAILED — no valid edges")
        return None
    print(f"      Pass 2 OK — {len(edges)} edges")

    # Pass 3 — marginals for ALL nodes
    p3_tokens = min(4000, max(1500, len(nodes) * 150))
    p3 = call_claude(client, PASS3_SYSTEM, pass3_prompt(nodes, text),
                     model, max_tokens=p3_tokens)
    if not p3 or not p3.get("marginals"):
        print("      [Pass 3] FAILED")
        return None
    marginals = validate_marginals(p3["marginals"], nodes)
    print(f"      Pass 3 OK — {len(marginals)} marginals")

    # Pass 4 — joint probabilities
    use_chunks = len(edges) > chunk_size
    if use_chunks:
        raw_joints = pass4_chunked(nodes, edges, text, client, model, chunk_size)
    else:
        max_tok    = pass4_tokens_needed(len(edges))
        p4         = call_claude(client, PASS4_SYSTEM,
                                 pass4_prompt(nodes, edges, text),
                                 model, max_tokens=max_tok)
        raw_joints = p4["joints"] if p4 and "joints" in p4 else []

    if not raw_joints:
        print("      [Pass 4] FAILED")
        return None
    joints = validate_joints(raw_joints, nodes, edges)
    print(f"      Pass 4 OK — {len(joints)} joint tables")

    return {
        "nodes":             nodes,
        "edges":             edges,
        "topological_order": topo,
        "marginals":         marginals,   # {node_name: {state: prob}}
        "joints":            joints,      # [{parent, child, table}]
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Wikipedia helpers  — robust versions
# ─────────────────────────────────────────────────────────────────────────────

def safe_wiki_search(query: str,
                     results:     int   = 80,
                     max_retries: int   = 4,
                     base_delay:  float = 5.0) -> list:
    """
    Wrapper around wikipedia.search() with exponential-backoff retry.
    Returns an empty list instead of crashing on bad/empty API responses.
    """
    for attempt in range(max_retries):
        try:
            found = wikipedia.search(query, results=results)
            return found if found else []

        except req_lib.exceptions.JSONDecodeError:
            # Wikipedia API returned an empty or non-JSON body
            wait = base_delay * (2 ** attempt)
            print(f"      [Wiki search] Empty/bad JSON for '{query}' "
                  f"(attempt {attempt+1}/{max_retries}) — retrying in {wait:.0f}s")
            time.sleep(wait)

        except req_lib.exceptions.ConnectionError as exc:
            wait = base_delay * (2 ** attempt)
            print(f"      [Wiki search] Connection error: {exc} — "
                  f"retrying in {wait:.0f}s")
            time.sleep(wait)

        except req_lib.exceptions.Timeout:
            wait = base_delay * (2 ** attempt)
            print(f"      [Wiki search] Timeout for '{query}' — "
                  f"retrying in {wait:.0f}s")
            time.sleep(wait)

        except Exception as exc:
            # Unknown error — log and give up immediately
            print(f"      [Wiki search] Unexpected error for '{query}': {exc}")
            return []

    print(f"      [Wiki search] All {max_retries} retries exhausted for '{query}' — skipping")
    return []


def safe_fetch_summary(title:       str,
                       max_retries: int   = 4,
                       base_delay:  float = 3.0):
    """
    Wrapper around wikipedia.page() with retry logic.
    Returns (summary_str, url_str) or (None, None) on failure.
    """
    for attempt in range(max_retries):
        try:
            page    = wikipedia.page(title, auto_suggest=False)
            summary = page.summary.strip()
            words   = len(summary.split())
            if words < 80 or words > 800:
                return None, None
            return summary, page.url

        except wikipedia.exceptions.DisambiguationError as exc:
            # Try the first unambiguous option
            try:
                page    = wikipedia.page(exc.options[0], auto_suggest=False)
                summary = page.summary.strip()
                words   = len(summary.split())
                if words < 80 or words > 800:
                    return None, None
                return summary, page.url
            except Exception:
                return None, None

        except wikipedia.exceptions.PageError:
            return None, None          # page doesn't exist — no point retrying

        except req_lib.exceptions.JSONDecodeError:
            wait = base_delay * (2 ** attempt)
            print(f"      [Wiki fetch] Empty/bad JSON for '{title}' "
                  f"(attempt {attempt+1}/{max_retries}) — retrying in {wait:.0f}s")
            time.sleep(wait)

        except req_lib.exceptions.ConnectionError as exc:
            wait = base_delay * (2 ** attempt)
            print(f"      [Wiki fetch] Connection error: {exc} — "
                  f"retrying in {wait:.0f}s")
            time.sleep(wait)

        except req_lib.exceptions.Timeout:
            wait = base_delay * (2 ** attempt)
            print(f"      [Wiki fetch] Timeout for '{title}' — "
                  f"retrying in {wait:.0f}s")
            time.sleep(wait)

        except Exception as exc:
            print(f"      [Wiki fetch] Unexpected error for '{title}': {exc}")
            return None, None

    print(f"      [Wiki fetch] All {max_retries} retries exhausted for '{title}' — skipping")
    return None, None


def collect_titles(domain: str, seed_categories: list, target: int) -> list:
    """
    Collects Wikipedia article titles for a domain.
    Robust to network errors and empty API responses.
    """
    titles = set()

    for cat in seed_categories:
        if len(titles) >= target * 3:
            print(f"  Reached {len(titles)} titles, stopping category search.")
            break

        results = safe_wiki_search(cat, results=80)
        for t in results:
            if ":" in t:          # skip meta-pages like "Category:..." etc.
                continue
            titles.add(t)

        time.sleep(0.5)           # polite delay between searches

    titles = list(titles)
    random.shuffle(titles)
    return titles


# ─────────────────────────────────────────────────────────────────────────────
#  Domain definitions
# ─────────────────────────────────────────────────────────────────────────────

DOMAINS = {
    "economics":   ["Economic crises","Inflation","Unemployment","Financial crises","Income inequality"],
    "environment": ["Effects of climate change", "Deforestation",
                    "Natural disasters", "Air pollution", "Water scarcity"],
    "geopolitics": ["Wars by cause", "Economic sanctions",
                    "Political crises", "Refugee crises", "Revolutions"],
    "technology":  ["Effects of technology on society", "Automation",
                    "Artificial intelligence", "Social media", "Cybersecurity"],
    "society":     ["Poverty", "Public health", "Social inequality",
                    "Crime", "Education"],
}


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def run_extractor(
    output_path: str   = "bn_marginal_raw.jsonl",
    api_key:     str   = None,
    per_domain:  int   = 10,
    model:       str   = os.getenv("EXTRACTION_MODEL", "extraction-model"),
    delay:       float = 0.5,
    chunk_size:  int   = 10,
):
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        print("ERROR: provide --api-key or set ANTHROPIC_API_KEY")
        return

    wikipedia.set_lang("en")

    client      = anthropic.Anthropic(api_key=key)
    n_written   = 0
    n_skipped   = 0
    seen_titles = set()

    print("Wikipedia BN extractor  (marginal approach)")
    print(f"  Domains:    {list(DOMAINS.keys())}")
    print(f"  Per domain: {per_domain}")
    print(f"  Model:      {model}")
    print(f"  Output:     {output_path}")
    print("-" * 55)

    with open(output_path, "w") as out:
        for domain, seed_cats in DOMAINS.items():
            print(f"\n[{domain}] collecting titles...")
            print(f"  Seed categories: {seed_cats}")
            print(f"  Target per domain: {per_domain}")

            titles = collect_titles(domain, seed_cats, per_domain)
            print(f"  Titles collected: {len(titles)}")

            written = 0

            for title in titles:
                if written >= per_domain:
                    break
                if title in seen_titles:
                    continue

                # ── fetch article ──────────────────────────────────────────
                summary, url = safe_fetch_summary(title)
                if summary is None:
                    continue

                seen_titles.add(title)
                print(f"  [{domain}] {title[:55]}")

                # ── extract BN ─────────────────────────────────────────────
                bn = extract_bn_marginal(summary, client, model, chunk_size)
                if bn is None:
                    n_skipped += 1
                    continue

                record = {
                    "id":                f"wiki_{n_written:05d}",
                    "title":             title,
                    "domain":            domain,
                    "url":               url,
                    "text":              summary,
                    "real":              1,
                    "nodes":             bn["nodes"],
                    "edges":             bn["edges"],
                    "topological_order": bn["topological_order"],
                    "marginals":         bn["marginals"],
                    "joints":            bn["joints"],
                    "n_nodes":           len(bn["nodes"]),
                    "n_edges":           len(bn["edges"]),
                }
                out.write(json.dumps(record) + "\n")
                out.flush()          # flush after every record — safe against crashes
                n_written += 1
                written   += 1

                if written % 20 == 0:
                    print(f"  [{domain}] {written}/{per_domain} "
                          f"(total={n_written})")

                time.sleep(delay)

    print(f"\nDone.  Written={n_written}  Skipped={n_skipped}")
    print(f"Output: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Extract BNs using marginal + joint approach."
    )
    ap.add_argument("--api-key",    default="",
                    help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    ap.add_argument("--output",     default="bn_marginal_raw.jsonl")
    ap.add_argument("--per-domain", type=int,   default=10)
    ap.add_argument("--model",      default=os.getenv("EXTRACTION_MODEL", "extraction-model"))
    ap.add_argument("--delay",      type=float, default=0.5)
    ap.add_argument("--chunk-size", type=int,   default=10,
                    help="Max edges per Pass 4 chunk (default: 10)")
    args = ap.parse_args()

    run_extractor(
        output_path = args.output,
        api_key     = args.api_key,
        per_domain  = args.per_domain,
        model       = args.model,
        delay       = args.delay,
        chunk_size  = args.chunk_size,
    )