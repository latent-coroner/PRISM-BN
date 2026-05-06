"""
llm_judge_llama_4pass.py — 4-pass Bayesian Network extraction & evaluation
Pass 1: Extract nodes        → Node F1, % nodes matched, extra nodes
Pass 2: Extract states       → State F1, % states matched
Pass 3: Extract edges        → % correct, % spurious connections
Pass 4: Extract CPD matrices → KL divergence

Extraction model : Extraction Model  (via Anthropic SDK)
Judge model      : Judge Model (via HuggingFace InferenceClient / Groq)
"""

import os, re, json, time, math, argparse, csv
from collections import defaultdict
import anthropic
from huggingface_hub import InferenceClient

EXTRACTION_MODEL = os.getenv("EXTRACTION_MODEL", "extraction-model")
JUDGE_MODEL      = os.getenv("JUDGE_MODEL", "judge-model")
HF_TOKEN = os.getenv("HF_TOKEN")

NODE_THRESHOLD  = 0.5
STATE_THRESHOLD = 0.5
LAPLACE_EPS     = 1e-6
MAX_RETRIES     = 3
DELAY           = 0.3

# ── Pass 1 prompts ────────────────────────────────────────────────────────────

PASS1_SYSTEM = """You are an expert at identifying nodes (variables) in Bayesian Networks from natural language.
Extract only the node names. Return ONLY valid JSON, no explanation."""

def pass1_prompt(text):
    return f"""Extract all Bayesian Network nodes (variables) from this text.

Text:
\"\"\"{text}\"\"\"

Return ONLY:
{{"nodes": ["node1", "node2", "node3"]}}

Rules:
- List every variable/node mentioned as a random variable
- Use exact names from the text
- Do not include states, only node names"""

# ── Pass 2 prompts ────────────────────────────────────────────────────────────

PASS2_SYSTEM = """You are an expert at identifying states (possible values) of Bayesian Network nodes from natural language.
For each node given, extract all possible states. Always include a 'None' state for every node.
Return ONLY valid JSON, no explanation."""

def pass2_prompt(text, nodes):
    return f"""For each Bayesian Network node listed, extract all possible states from this text.

Text:
\"\"\"{text}\"\"\"

Nodes: {json.dumps(nodes)}

Return ONLY:
{{
  "node_states": [
    {{"node": "node name", "states": ["state1", "state2", "None"]}}
  ]
}}

Rules:
- Include every node from the list above
- List all states mentioned in the text for each node
- Always include "None" as a state for every node"""

# ── Pass 3 prompts ────────────────────────────────────────────────────────────

PASS3_SYSTEM = """You are an expert at identifying directed causal relationships between Bayesian Network nodes.
Given nodes and their states, extract which nodes causally influence which others.
Return ONLY valid JSON, no explanation."""

def pass3_prompt(text, node_states):
    nodes_info = "\n".join(
        f"  - {ns['node']}: {json.dumps(ns['states'])}"
        for ns in node_states
    )
    return f"""Identify directed causal edges (A → B means A causes/influences B) between these nodes.

Text:
\"\"\"{text}\"\"\"

Available nodes and their states:
{nodes_info}

Return ONLY:
{{
  "edges": [
    {{"parent": "cause node", "child": "effect node"}}
  ]
}}

Rules:
- Only use node names from the list above
- Only include edges supported by the text
- An edge parent→child means the parent causally influences the child"""

# ── Pass 4 prompts ────────────────────────────────────────────────────────────

PASS4_SYSTEM = """You are an expert at extracting Conditional Probability Distributions (CPDs) from natural language.
Given a parent-child node pair with their states, produce the CPD matrix.
Rows = child states, columns = parent states. Each column must sum to 1.0.
Return ONLY valid JSON, no explanation."""

def pass4_prompt(text, parent_node, parent_states, child_node, child_states):
    n_rows = len(child_states)
    n_cols = len(parent_states)
    return f"""Extract the CPD matrix for this causal relationship from the text.

Text:
\"\"\"{text}\"\"\"

Parent "{parent_node}" states (columns): {json.dumps(parent_states)}
Child  "{child_node}"  states (rows):    {json.dumps(child_states)}

Return ONLY:
{{
  "parent": "{parent_node}",
  "child":  "{child_node}",
  "matrix": <{n_rows}x{n_cols} list of lists, each column sums to 1.0>
}}

Rules:
- Matrix is {n_rows} rows x {n_cols} columns
- Row order matches child states exactly: {json.dumps(child_states)}
- Column order matches parent states exactly: {json.dumps(parent_states)}
- Every column must sum to 1.0"""

# ── Judge prompts (binary scoring) ───────────────────────────────────────────

NODE_JUDGE_SYSTEM = """Match predicted Bayesian Network node names to ground truth node names.
Scoring: 1=exact match or very minor variation (e.g., singular/plural), 0=no meaningful match.
One-to-one matching only. Return ONLY valid JSON."""

def node_judge_prompt(gt_names, pred_names):
    return f"""Match each predicted node name to a ground truth node name.

Ground truth: {json.dumps(gt_names)}
Predicted:    {json.dumps(pred_names)}

Return ONLY:
{{
  "matches": [
    {{"predicted": "pred name", "ground_truth": "gt name", "score": 1}}
  ],
  "unmatched_predicted": [],
  "unmatched_ground_truth": []
}}

Rules:
- score=1 if same meaning or clear paraphrase, score=0 if no meaningful match
- Only include matches with score=1
- One predicted name can match at most one ground truth name"""

STATE_JUDGE_SYSTEM = """Match predicted states to ground truth states for a Bayesian Network node.
"None" always maps to "None".
Scoring: 1=exact match or very minor variation (e.g., singular/plural), 0=no meaningful match.
One-to-one matching only. Return ONLY valid JSON."""

def state_judge_prompt(gt_node, pred_node, gt_states, pred_states):
    return f"""Match predicted states to ground truth states for node "{gt_node}".

Ground truth states: {json.dumps(gt_states)}
Predicted states:    {json.dumps(pred_states)}

Return ONLY:
{{
  "state_matches": [
    {{"predicted": "pred state", "ground_truth": "gt state", "score": 1}}
  ],
  "unmatched_predicted": [],
  "unmatched_ground_truth": []
}}

Rules:
- score=1 if same meaning or clear paraphrase, score=0 if no meaningful match
- Only include matches with score=1
- "None" always maps to "None" if both lists contain it"""

# ── API helpers ───────────────────────────────────────────────────────────────

def extract_json_robust(raw):
    raw   = re.sub(r'^```[a-zA-Z]*\n?', '', raw.strip())
    raw   = re.sub(r'\n?```$', '', raw).strip()
    start = raw.find('{'); end = raw.rfind('}')
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON found")
    return json.loads(raw[start:end+1])

def call_claude(client, system, user, max_tokens=2000):
    """Extraction calls — Extraction Model via Anthropic SDK."""
    for attempt in range(MAX_RETRIES):
        try:
            msg = client.messages.create(
                model=EXTRACTION_MODEL, max_tokens=max_tokens, system=system,
                messages=[{"role": "user", "content": user}])
            return extract_json_robust(msg.content[0].text.strip())
        except (json.JSONDecodeError, ValueError) as e:
            print(f"      [JSON err {attempt+1}]: {e}"); time.sleep(1)
        except anthropic.RateLimitError:
            print("      [Rate limit — 60s]"); time.sleep(60)
        except Exception as e:
            print(f"      [API err {attempt+1}]: {e}"); time.sleep(2)
    return None

def call_llama(hf_client, system, user, max_tokens=2000):
    """Judge calls — Judge Model via HuggingFace InferenceClient (Groq)."""
    for attempt in range(MAX_RETRIES):
        try:
            completion = hf_client.chat.completions.create(
                model=JUDGE_MODEL,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
            )
            text = completion.choices[0].message.content.strip()
            return extract_json_robust(text)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"      [JSON err {attempt+1}]: {e}"); time.sleep(1)
        except Exception as e:
            err = str(e)
            if "rate" in err.lower() or "429" in err:
                print("      [Rate limit — 60s]"); time.sleep(60)
            else:
                print(f"      [API err {attempt+1}]: {e}"); time.sleep(2)
    return None

# ── Metric helpers ────────────────────────────────────────────────────────────

def f1_prf(n_corr, n_gt, n_pred):
    prec = n_corr / n_pred if n_pred > 0 else 0.0
    rec  = n_corr / n_gt   if n_gt   > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return round(f1, 4), round(prec, 4), round(rec, 4)

def kl_div(p, q):
    p  = [0.0 if v is None else float(v) for v in p]
    q  = [0.0 if v is None else float(v) for v in q]
    pt = sum(p); p = [v/pt for v in p] if pt > 0 else [1.0/len(p)] * len(p)
    qs = [max(v, 0.0) + LAPLACE_EPS for v in q]; qn = [v/sum(qs) for v in qs]
    return sum(pi * math.log(pi/qi) for pi, qi in zip(p, qn) if pi > 0)

def col_kl(pred_matrix, gt_matrix, n_gt_parent_states, n_gt_child_states):
    """Average KL divergence over parent-state columns, aligned to GT dimensions."""
    nr  = len(pred_matrix)
    nc  = len(pred_matrix[0]) if pred_matrix else 0
    gt_nr = len(gt_matrix)
    kls = []
    for j in range(n_gt_parent_states):
        if j >= nc:
            break
        gt_col = [float(gt_matrix[r][j]) if r < gt_nr and j < len(gt_matrix[r]) and gt_matrix[r][j] is not None else 0.0
                  for r in range(n_gt_child_states)]
        pr_col = [float(pred_matrix[r][j]) if j < len(pred_matrix[r]) and pred_matrix[r][j] is not None else 0.0
                  for r in range(nr)]
        if len(pr_col) > len(gt_col):   pr_col = pr_col[:len(gt_col)]
        elif len(pr_col) < len(gt_col): pr_col += [0.0] * (len(gt_col) - len(pr_col))
        cs     = sum(pr_col)
        pr_col = [v/cs for v in pr_col] if cs > 0 else [1.0/len(gt_col)] * len(gt_col)
        kls.append(kl_div(gt_col, pr_col))
    return round(sum(kls)/len(kls), 6) if kls else None

# ── Pass 1 eval ───────────────────────────────────────────────────────────────

def eval_pass1(judge_result, n_gt, n_pred):
    valid = [m for m in judge_result.get("matches", []) if m.get("score", 0) > 0 and "predicted" in m and "ground_truth" in m]

    by_gt = defaultdict(list)
    for m in valid:
        by_gt[m["ground_truth"]].append(m)

    canonical_matches, extras_map = [], {}
    for gt_name, matches in by_gt.items():
        canonical_matches.append(matches[0])
        if len(matches) > 1:
            extras_map[gt_name] = [m["predicted"] for m in matches[1:]]

    n_extra    = sum(len(v) for v in extras_map.values())
    n_corr     = len(canonical_matches)
    n_pred_eff = max(n_pred - n_extra, n_corr)
    f1, prec, rec = f1_prf(n_corr, n_gt, n_pred_eff)

    metrics = {
        "node_f1": f1, "node_precision": prec, "node_recall": rec,
        "pct_nodes_matched": round(n_corr / n_gt * 100, 1) if n_gt > 0 else 0.0,
        "n_matched_nodes": n_corr, "n_gt_nodes": n_gt, "n_pred_nodes": n_pred,
        "n_extra_nodes": n_extra,
    }
    node_match_map = {m["predicted"]: m["ground_truth"] for m in canonical_matches}
    return metrics, node_match_map, extras_map

# ── Pass 2 eval ───────────────────────────────────────────────────────────────

def eval_pass2_node(judge_result, gt_states, pred_states):
    valid  = [m for m in judge_result.get("state_matches", []) if m.get("score", 0) > 0 and "predicted" in m and "ground_truth" in m]
    n_corr = len(valid)
    f1, prec, rec = f1_prf(n_corr, len(gt_states), len(pred_states))
    metrics   = {"state_f1": f1, "state_precision": prec, "state_recall": rec,
                 "n_matched": n_corr, "n_gt": len(gt_states), "n_pred": len(pred_states)}
    state_map = {m["predicted"]: m["ground_truth"] for m in valid}
    return metrics, state_map

# ── Pass 3 eval ───────────────────────────────────────────────────────────────

def eval_pass3(gt_edges, pred_edges, node_match_map):
    gt_set = {(e["parent"], e["child"]) for e in gt_edges}
    pred_mapped = []
    for e in pred_edges:
        pg = node_match_map.get(e.get("parent", ""))
        cg = node_match_map.get(e.get("child",  ""))
        if pg and cg:
            pred_mapped.append((pg, cg))
    pred_set = set(pred_mapped)
    correct  = gt_set & pred_set
    spurious = pred_set - gt_set
    n_gt, n_pred, n_corr = len(gt_set), len(pred_set), len(correct)
    f1, prec, rec = f1_prf(n_corr, n_gt, n_pred)
    return {
        "edge_f1": f1, "edge_precision": prec, "edge_recall": rec,
        "pct_correct_edges":  round(n_corr/n_gt*100,          1) if n_gt   > 0 else 0.0,
        "pct_spurious_edges": round(len(spurious)/n_pred*100,  1) if n_pred > 0 else 0.0,
        "n_correct_edges": n_corr, "n_gt_edges": n_gt, "n_pred_edges": n_pred,
        "n_spurious_edges": len(spurious),
        "correct_edges": list(correct),
    }

# ── Process one subgraph ──────────────────────────────────────────────────────

def process_subgraph(sg_id, sg, claude_client, llama_client):
    result = {
        "id": sg_id, "title": sg.get("title", ""), "domain": sg.get("domain", ""),
        "n_nodes": sg.get("n_nodes", 0), "n_edges": sg.get("n_edges", 0),
        "n_total_states": None, "error": None,
        # Pass 1
        "node_f1": None, "node_precision": None, "node_recall": None,
        "pct_nodes_matched": None, "n_matched_nodes": None,
        "n_gt_nodes": None, "n_pred_nodes": None, "n_extra_nodes": None,
        # Pass 2
        "state_f1": None, "state_precision": None, "state_recall": None,
        "pct_states_matched": None,
        # Pass 3
        "edge_f1": None, "edge_precision": None, "edge_recall": None,
        "pct_correct_edges": None, "pct_spurious_edges": None,
        "n_correct_edges": None, "n_gt_edges": None, "n_pred_edges": None,
        # Pass 4
        "cpd_kl": None, "cpd_kl_symmetric": None, "n_cpd_evaluated": None,
    }

    text = sg.get("generated_text", "")
    if not text:
        result["error"] = "no_generated_text"; return result

    gt_nodes = sg.get("nodes", {})
    gt_names = list(gt_nodes.keys())
    result["n_total_states"] = sum(len(info.get("states", [])) for info in gt_nodes.values())
    gt_edges = [{"parent": p, "child": c}
                for c, info in gt_nodes.items()
                for p in info.get("parents", {}).keys()]

    # ── Pass 1: extract nodes (Claude) ───────────────────────────────────────
    print("    Pass 1: nodes...")
    p1 = call_claude(claude_client, PASS1_SYSTEM, pass1_prompt(text),
                     max_tokens=min(1000, 200 + len(gt_names) * 50))
    time.sleep(DELAY)
    if not p1:
        result["error"] = "pass1_failed"; return result

    pred_names = p1.get("nodes", [])
    if not pred_names:
        result["error"] = "no_nodes_extracted"; return result

    # Judge nodes (Llama)
    nj = call_llama(llama_client, NODE_JUDGE_SYSTEM,
                    node_judge_prompt(gt_names, pred_names),
                    max_tokens=min(2000, 500 + len(gt_names)*60 + len(pred_names)*60))
    time.sleep(DELAY)
    if not nj:
        result["error"] = "node_judge_failed"; return result

    node_metrics, node_match_map, extras_map = eval_pass1(nj, len(gt_names), len(pred_names))
    result.update(node_metrics)
    print(f"      GT nodes:   {gt_names}")
    print(f"      Pred nodes: {pred_names}")
    print(f"      node_f1={node_metrics['node_f1']}  "
          f"matched={node_metrics['n_matched_nodes']}/{node_metrics['n_gt_nodes']}  "
          f"extras={node_metrics['n_extra_nodes']}")

    if not node_match_map:
        result["error"] = "no_node_matches"; return result

    # ── Pass 2: extract states (Claude) ──────────────────────────────────────
    print("    Pass 2: states...")
    matched_pred_nodes = list(node_match_map.keys())
    all_extra_names    = [n for ns in extras_map.values() for n in ns]
    all_pred_for_p2    = matched_pred_nodes + all_extra_names

    p2 = call_claude(claude_client, PASS2_SYSTEM,
                     pass2_prompt(text, all_pred_for_p2),
                     max_tokens=min(3000, 500 + len(all_pred_for_p2) * 200))
    time.sleep(DELAY)

    pred_node_states = {}
    if p2:
        for ns in p2.get("node_states", []):
            states = ns.get("states", [])
            if "None" not in states and "none" not in [s.lower() for s in states]:
                states = states + ["None"]
            pred_node_states[ns["node"]] = states

    for gt_name, extra_names in extras_map.items():
        canonical = next((p for p, g in node_match_map.items() if g == gt_name), None)
        if canonical is None:
            continue
        combined = list(pred_node_states.get(canonical, []))
        for extra in extra_names:
            for s in pred_node_states.get(extra, []):
                if s not in combined:
                    combined.append(s)
        if "None" not in combined:
            combined.append("None")
        pred_node_states[canonical] = combined

    per_node_state = []
    for pred_name, gt_name in node_match_map.items():
        gt_states   = gt_nodes.get(gt_name, {}).get("states", [])
        pred_states = pred_node_states.get(pred_name, [])
        if not gt_states:
            continue
        if not pred_states:
            print(f"        {gt_name}: GT={gt_states}, Pred={pred_states}")
            per_node_state.append({"gt_node": gt_name, "pred_node": pred_name,
                "state_f1": 0.0, "state_precision": 0.0, "state_recall": 0.0,
                "n_matched": 0, "n_gt": len(gt_states), "n_pred": 0})
            continue

        print(f"        {gt_name}: GT={gt_states}, Pred={pred_states}")
        # Judge states (Llama)
        sj = call_llama(llama_client, STATE_JUDGE_SYSTEM,
                        state_judge_prompt(gt_name, pred_name, gt_states, pred_states),
                        max_tokens=min(2000, 400 + len(gt_states)*60 + len(pred_states)*60))
        time.sleep(DELAY)

        if not sj:
            exact = set(s.lower() for s in gt_states) & set(s.lower() for s in pred_states)
            nc    = len(exact)
            f1, prec, rec = f1_prf(nc, len(gt_states), len(pred_states))
            m = {"state_f1": f1, "state_precision": prec, "state_recall": rec,
                 "n_matched": nc, "n_gt": len(gt_states), "n_pred": len(pred_states)}
        else:
            m, _ = eval_pass2_node(sj, gt_states, pred_states)

        m["gt_node"] = gt_name; m["pred_node"] = pred_name
        per_node_state.append(m)

    if per_node_state:
        def avg(k):
            v = [d[k] for d in per_node_state if d.get(k) is not None]
            return round(sum(v)/len(v), 4) if v else None
        result["state_f1"]        = avg("state_f1")
        result["state_precision"] = avg("state_precision")
        result["state_recall"]    = avg("state_recall")
        tot_gt  = sum(d["n_gt"]      for d in per_node_state)
        tot_mat = sum(d["n_matched"] for d in per_node_state)
        result["pct_states_matched"] = round(tot_mat/tot_gt*100, 1) if tot_gt > 0 else 0.0
        print(f"      state_f1={result['state_f1']}  matched={tot_mat}/{tot_gt}")

    # ── Pass 3: extract edges (Claude) ────────────────────────────────────────
    print("    Pass 3: edges...")
    node_states_for_prompt = []
    for pred_name in matched_pred_nodes:
        states = pred_node_states.get(pred_name, ["None"])
        if "None" not in states and "none" not in [s.lower() for s in states]:
            states = states + ["None"]
        node_states_for_prompt.append({"node": pred_name, "states": states})

    p3 = call_claude(claude_client, PASS3_SYSTEM,
                     pass3_prompt(text, node_states_for_prompt),
                     max_tokens=min(2000, 400 + len(matched_pred_nodes) * 100))
    time.sleep(DELAY)

    pred_edges   = p3.get("edges", []) if p3 else []
    edge_metrics = eval_pass3(gt_edges, pred_edges, node_match_map)
    result.update({k: edge_metrics[k] for k in edge_metrics if k != "correct_edges"})
    print(f"      edge_f1={edge_metrics['edge_f1']}  "
          f"correct={edge_metrics['n_correct_edges']}/{edge_metrics['n_gt_edges']}  "
          f"spurious={edge_metrics['n_spurious_edges']}")

    # ── Pass 4: CPD matrix for each correctly predicted edge (Claude) ─────────
    correct_edges = edge_metrics.get("correct_edges", [])
    if not correct_edges:
        print("      Pass 4: no correct edges — skipping")
        return result

    print(f"    Pass 4: CPDs for {len(correct_edges)} correct edges...")
    gt_to_pred = {v: k for k, v in node_match_map.items()}
    kl_values  = []
    kl_sym_values = []

    for (gt_parent, gt_child) in correct_edges:
        pred_parent = gt_to_pred.get(gt_parent, gt_parent)
        pred_child  = gt_to_pred.get(gt_child,  gt_child)

        pred_p_states = pred_node_states.get(pred_parent, [])
        pred_c_states = pred_node_states.get(pred_child,  [])
        if not pred_p_states or not pred_c_states:
            continue

        gt_child_info  = gt_nodes.get(gt_child, {})
        gt_parent_info = gt_child_info.get("parents", {}).get(gt_parent, {})
        gt_matrix      = gt_parent_info.get("cpd_matrix", [])
        gt_p_states    = gt_parent_info.get("parent_states", [])
        gt_c_states    = gt_child_info.get("states", [])
        if not gt_matrix:
            continue

        p4 = call_claude(claude_client, PASS4_SYSTEM,
                         pass4_prompt(text, pred_parent, pred_p_states,
                                      pred_child, pred_c_states),
                         max_tokens=min(2000, 400 + len(pred_p_states)*len(pred_c_states)*20))
        time.sleep(DELAY)

        if not p4:
            continue

        kl = col_kl(p4.get("matrix", []), gt_matrix, len(gt_p_states), len(gt_c_states))
        kl_rev = col_kl(gt_matrix, p4.get("matrix", []), len(gt_p_states), len(gt_c_states))
        if kl is not None and kl_rev is not None:
            kl_sym = (kl + kl_rev) / 2.0
            kl_values.append(kl)
            kl_sym_values.append(kl_sym)
            print(f"        {gt_parent}→{gt_child}  KL={kl:.4f}  KL-sym={kl_sym:.4f}")

    if kl_values:
        result["cpd_kl"]          = round(sum(kl_values)/len(kl_values), 6)
        result["n_cpd_evaluated"] = len(kl_values)
    if kl_sym_values:
        result["cpd_kl_symmetric"] = round(sum(kl_sym_values)/len(kl_sym_values), 6)
    print(f"      cpd_kl={result['cpd_kl']}  cpd_kl_symmetric={result['cpd_kl_symmetric']}")

    return result

# ── Aggregate ─────────────────────────────────────────────────────────────────

def aggregate(results):
    def ms(vals):
        vals = [v for v in vals if v is not None]
        if not vals: return None, None
        m = sum(vals)/len(vals)
        return round(m, 4), round(math.sqrt(sum((v-m)**2 for v in vals)/len(vals)), 4)

    keys = ["node_f1", "node_recall", "pct_nodes_matched",
            "state_f1", "state_recall", "pct_states_matched",
            "edge_f1", "edge_recall", "pct_correct_edges", "pct_spurious_edges",
            "cpd_kl", "cpd_kl_symmetric", "n_extra_nodes"]
    agg  = {"total": len(results), "extraction_model": EXTRACTION_MODEL,
            "judge_model": JUDGE_MODEL}
    for k in keys:
        m, s = ms([r.get(k) for r in results])
        agg[f"{k}_mean"] = m; agg[f"{k}_std"] = s
    return agg

# ── CSV ───────────────────────────────────────────────────────────────────────

def write_csv(results, path):
    fields = ["id", "title", "domain", "n_nodes", "n_edges",
              "node_f1", "node_precision", "node_recall", "pct_nodes_matched",
              "n_matched_nodes", "n_gt_nodes", "n_pred_nodes", "n_extra_nodes",
              "state_f1", "state_precision", "state_recall", "pct_states_matched",
              "edge_f1", "edge_precision", "edge_recall",
              "pct_correct_edges", "pct_spurious_edges",
              "n_correct_edges", "n_gt_edges", "n_pred_edges",
              "cpd_kl", "n_cpd_evaluated", "error"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results: w.writerow(r)
    print(f"  CSV: {path}")

# ── Result analysis & plots ───────────────────────────────────────────────────

def run_result_analysis(results, out_dir="anonymized_result_analysis"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    BG = "#FFFFFF"
    C  = {"node": "#2C6E8A", "state": "#7B52AB", "edge": "#4A9D8F",
          "cpd":  "#E07B39", "extra": "#C0392B"}

    def grp(results, key):
        g = defaultdict(list)
        for r in results:
            v = r.get(key)
            if v is not None:
                g[int(v)].append(r)
        return dict(sorted(g.items()))

    def bxp(ax, grps, ks, metric, color, ylabel, title, pct=False):
        data, labels = [], []
        for k in ks:
            v = [r[metric] for r in grps[k] if r.get(metric) is not None]
            if pct:
                v = [x * 100 if x <= 1.0 else x for x in v]
            if v:
                data.append(v)
                labels.append(f"{k}\n(n=5054)")
        if not data:
            ax.set_visible(False); return
        bp = ax.boxplot(data, labels=labels, patch_artist=True,
                        medianprops={"color": "white", "linewidth": 2},
                        whiskerprops={"color": "#888", "linewidth": 1},
                        capprops={"color": "#888", "linewidth": 1},
                        flierprops={"marker": "o", "markerfacecolor": color,
                                    "markersize": 3, "alpha": 0.5,
                                    "markeredgewidth": 0}, zorder=3)
        for p in bp["boxes"]:
            p.set_facecolor(color); p.set_alpha(0.78)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold", color="#1A3A4A", pad=7)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
        ax.set_facecolor("#FAFAFA")
        ax.spines[["top", "right"]].set_visible(False)
        if pct:
            ax.set_ylim(-5, 108)
            ax.axhline(50, color="#bbb", lw=0.7, ls=":")

    nd = len(results)
    gn = grp(results, "n_nodes")
    kn = sorted(gn.keys())

    X_LABEL = "Number of Unique Nodes in Ground Truth (3–8)"
    TAG = "[Extraction Model + Judge Model]"

    plot_specs = [
        ("pct_nodes_matched",  C["node"],  "% Nodes Matched",
         f"% Nodes Correctly Predicted vs Unique Nodes  {TAG}",
         "plot1_pct_nodes_vs_nodes.png"),
        ("pct_states_matched", C["state"], "% States Matched",
         f"% States Correctly Predicted vs Unique Nodes  {TAG}",
         "plot2_pct_states_vs_nodes.png"),
        ("node_f1",            C["node"],  "Node F1",
         f"Node F1 vs Unique Nodes  {TAG}",
         "plot3_node_f1_vs_nodes.png"),
        ("state_f1",           C["state"], "State F1",
         f"State F1 vs Unique Nodes  {TAG}",
         "plot4_state_f1_vs_nodes.png"),
    ]

    for metric, color, ylabel, title, fname in plot_specs:
        w = max(6, len(kn) * 1.4 + 2)
        fig, ax = plt.subplots(figsize=(w, 5))
        fig.patch.set_facecolor(BG)
        bxp(ax, gn, kn, metric, color, ylabel, title, pct=False)
        ax.set_xlabel(X_LABEL, fontsize=10)
        fig.suptitle(f"{title}  (n={nd})", fontsize=12, color="#555", y=1.02)
        plt.tight_layout(pad=1.2)
        plt.savefig(os.path.join(out_dir, fname), dpi=160,
                    bbox_inches="tight", facecolor=BG)
        plt.close()
        print(f"  Saved: {os.path.join(out_dir, fname)}")

    for metric, ylabel, fname in [
        ("pct_correct_edges",  "% Correct Connections",         "plot5_pct_correct_edges_vs_nodes.png"),
        ("cpd_kl",             "CPD-KL Divergence  (↓)",        "plot6_kl_vs_nodes.png"),
        ("cpd_kl_symmetric",   "CPD-KL-Symmetric Divergence  (↓)", "plot6b_kl_symmetric_vs_nodes.png"),
        ("pct_spurious_edges", "% Spurious Connections",        "plot7_pct_spurious_edges_vs_nodes.png"),
        ("n_extra_nodes",      "# Extra Predicted Nodes",       "plot8_extra_nodes_vs_nodes.png"),
    ]:
        color = C["cpd"] if "kl" in metric else C["extra"] if "extra" in metric else C["edge"]
        w = max(6, len(kn) * 1.4 + 2)
        fig, ax = plt.subplots(figsize=(w, 5))
        fig.patch.set_facecolor(BG)
        bxp(ax, gn, kn, metric, color, ylabel,
            f"{ylabel} vs Unique Nodes  {TAG}", pct=False)
        ax.set_xlabel(X_LABEL, fontsize=10)
        fig.suptitle(f"{ylabel}  (n={nd})  {TAG}", fontsize=12, color="#555", y=1.02)
        plt.tight_layout(pad=1.2)
        plt.savefig(os.path.join(out_dir, fname), dpi=160,
                    bbox_inches="tight", facecolor=BG)
        plt.close()
        print(f"  Saved: {os.path.join(out_dir, fname)}")

    def grouped_series(grps, metric):
        return {
            str(k): [r[metric] for r in v if r.get(metric) is not None]
            for k, v in grps.items()
        }

    plot_data = {
        "extraction_model": EXTRACTION_MODEL,
        "judge_model": JUDGE_MODEL,
        "n_total": nd,
        "x_axis": "n_nodes (3-8)",
        "plot1_pct_nodes_vs_nodes":         grouped_series(gn, "pct_nodes_matched"),
        "plot2_pct_states_vs_nodes":        grouped_series(gn, "pct_states_matched"),
        "plot3_node_f1_vs_nodes":           grouped_series(gn, "node_f1"),
        "plot4_state_f1_vs_nodes":          grouped_series(gn, "state_f1"),
        "plot5_pct_correct_edges_vs_nodes": grouped_series(gn, "pct_correct_edges"),
        "plot6_kl_vs_nodes":                grouped_series(gn, "cpd_kl"),
        "plot6b_kl_symmetric_vs_nodes":     grouped_series(gn, "cpd_kl_symmetric"),
        "plot7_pct_spurious_vs_nodes":      grouped_series(gn, "pct_spurious_edges"),
        "plot8_extra_nodes_vs_nodes":       grouped_series(gn, "n_extra_nodes"),
    }
    json_path = os.path.join(out_dir, "plot_data.json")
    with open(json_path, "w") as f:
        json.dump(plot_data, f, indent=2)
    print(f"  Plot data JSON: {json_path}")
    print(f"  [result_analysis] 9 plots + plot_data.json saved (n={nd})")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key",         default=os.environ.get("ANTHROPIC_API_KEY"), help="Anthropic API key for Claude")
    ap.add_argument("--input",           default="prism_bn.json")
    ap.add_argument("--output-json",     default="anonymized_eval_result.json")
    ap.add_argument("--output-csv",      default="anonymized_eval_result.csv")
    ap.add_argument("--output-analysis", default="anonymized_eval_result")
    ap.add_argument("--limit",           type=int, default=None)
    ap.add_argument("--watch",           action="store_true", help="Continuously reload input JSON and process new IDs")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: {args.input} not found"); return

    claude_client = anthropic.Anthropic(api_key=args.api_key)
    llama_client  = InferenceClient(api_key=HF_TOKEN)

    print(f"4-Pass BN Benchmark  (binary scoring: 1=match, 0=no match)")
    print(f"  Extraction: {EXTRACTION_MODEL}")
    print(f"  Judge:      {JUDGE_MODEL}")
    if args.watch:
        print(f"  Mode:       WATCH (continuously reload {args.input})")
    print("-" * 65)

    results = []
    done_ids = set()
    if os.path.exists(args.output_json):
        try:
            with open(args.output_json) as f:
                ckpt = json.load(f)
            results = ckpt.get("results", [])
            done_ids = {r["id"] for r in results if "id" in r}
            print(f"  [resume] loaded {len(done_ids)} completed samples from {args.output_json}")
        except Exception as e:
            print(f"  [resume] could not read checkpoint ({e}), starting fresh")

    total_processed = len(results)
    iteration = 0
    while True:
        iteration += 1
        data = json.loads(open(args.input).read())
        bns = data.get("bayesian_networks", {})
        if args.limit: bns = dict(list(bns.items())[:args.limit])

        new_ids = set(bns.keys()) - done_ids
        if not new_ids:
            if args.watch:
                print(f"  [watch {iteration}] No new IDs, waiting 10s... (done_ids: {len(done_ids)})")
                time.sleep(10)
                continue
            else:
                break

        if iteration == 1:
            print(f"  Subgraphs in input: {len(bns)}")
        else:
            print(f"  [watch {iteration}] Found {len(new_ids)} new IDs ({len(done_ids)} done so far)")

        for sg_id in new_ids:
            if sg_id in done_ids:
                continue
            sg = bns[sg_id]
            print(f"\n[{total_processed+1}] {sg_id} | {sg.get('domain','')} | {sg.get('title','')[:40]}")
            r = process_subgraph(sg_id, sg, claude_client, llama_client)
            results.append(r)
            done_ids.add(sg_id)
            total_processed += 1
            if r.get("error"):
                print(f"  FAIL — {r['error']}")
            else:
                print(f"  node_f1={r.get('node_f1')}  state_f1={r.get('state_f1')}  "
                      f"edge_f1={r.get('edge_f1')}  cpd_kl={r.get('cpd_kl')}  "
                      f"extra_nodes={r.get('n_extra_nodes')}")
            time.sleep(DELAY)

            if (total_processed % 10) == 0:
                print(f"  [checkpoint] saving after {total_processed} samples...")
                ckpt_agg = aggregate(results)
                with open(args.output_json, "w") as f:
                    json.dump({"extraction_model": EXTRACTION_MODEL, "judge_model": JUDGE_MODEL,
                               "aggregate": ckpt_agg, "results": results}, f, indent=2)
                write_csv(results, args.output_csv)

        if not args.watch:
            break
        print(f"  [watch] Finished batch, sleeping 10s before next reload...")
        time.sleep(10)

    if results:
        agg = aggregate(results)
        print(f"\n{'='*65}")
        print(f"  Node F1:         {agg['node_f1_mean']} ± {agg['node_f1_std']}")
        print(f"  % Nodes matched: {agg['pct_nodes_matched_mean']}")
        print(f"  Extra nodes:     {agg['n_extra_nodes_mean']} ± {agg['n_extra_nodes_std']}")
        print(f"  State F1:        {agg['state_f1_mean']} ± {agg['state_f1_std']}")
        print(f"  % States matched:{agg['pct_states_matched_mean']}")
        print(f"  Edge F1:         {agg['edge_f1_mean']} ± {agg['edge_f1_std']}")
        print(f"  % Correct edges: {agg['pct_correct_edges_mean']}")
        print(f"  % Spurious edges:{agg['pct_spurious_edges_mean']}")
        print(f"  CPD-KL:          {agg['cpd_kl_mean']} ± {agg['cpd_kl_std']}")
        print(f"  CPD-KL-Sym:      {agg['cpd_kl_symmetric_mean']} ± {agg['cpd_kl_symmetric_std']}")

        with open(args.output_json, "w") as f:
            json.dump({"extraction_model": EXTRACTION_MODEL, "judge_model": JUDGE_MODEL,
                       "aggregate": agg, "results": results}, f, indent=2)
        print(f"  JSON: {args.output_json}")
        write_csv(results, args.output_csv)

        print(f"\nGenerating result analysis plots → {args.output_analysis}/")
        run_result_analysis(results, out_dir=args.output_analysis)

if __name__ == "__main__":
    main()
