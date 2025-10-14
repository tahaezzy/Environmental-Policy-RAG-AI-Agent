# src/visualization.py
from src.config import logging, json, os, nx, List, Dict, Any, Optional
import textwrap 
from prettytable import PrettyTable

try:
    import matplotlib.pyplot as plt 
    logging.info("Pyplot loaded successfully")
except ImportError:
    logging.warning("Pyplot not available - graph features disabled")


def _to_display_text(x: Any, maxlen: int = 140) -> str:
    """Convert various types (str, list, dict) to a readable single-line string."""
    if x is None:
        return ""
    if isinstance(x, str):
        s = x
    elif isinstance(x, (list, tuple)):
        s = " | ".join([_to_display_text(i, maxlen) for i in x])
    elif isinstance(x, dict):
        # Prefer obvious text keys if present
        for k in ("text", "evidence", "quote", "content"):
            if k in x and x[k]:
                s = str(x[k])
                break
        else:
            # fallback: join pairs
            s = "; ".join(f"{k}:{v}" for k, v in x.items())
    else:
        s = str(x)
    s = " ".join(s.split())  # collapse whitespace
    if len(s) > maxlen:
        return s[: maxlen - 3] + "..."
    return s

def flatten_compliance_logs(log_path: str = "compliance_logs.jsonl") -> List[Dict[str, Any]]:
    """
    Read JSONL and return a flattened list of flag dicts with normalized fields:
    [
      {
        'file': '...', 'section_title': '...', 'issue': '...', 'evidence': '...', 
        'reg_ref': '...', 'severity': '...', 'confidence': 0.87, 'raw': {...}
      }, ...
    ]
    """
    if not os.path.exists(log_path):
        return []

    rows: List[Dict[str, Any]] = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except Exception:
                continue
            # Attempt to extract file / section context
            meta = entry.get("metadata") or {}
            file_name = meta.get("source") or entry.get("project_pdf") or entry.get("filename") or "unknown"
            section_title = entry.get("section_title") or meta.get("title") or entry.get("section") or None

            flags = entry.get("flags") or []
            for flag in flags:
                if not isinstance(flag, dict):
                    continue
                issue = _to_display_text(flag.get("issue"))
                evidence = _to_display_text(flag.get("evidence"), maxlen=300)
                reg_ref = _to_display_text(flag.get("reg_ref"), maxlen=200)
                severity = flag.get("severity") or flag.get("severity_level") or "medium"
                try:
                    confidence = float(flag.get("confidence", 0.0))
                except Exception:
                    confidence = 0.0
                rows.append({
                    "file": file_name,
                    "section_title": section_title,
                    "issue": issue,
                    "evidence": evidence,
                    "reg_ref": reg_ref,
                    "severity": severity,
                    "confidence": round(confidence, 3),
                    "raw_flag": flag,
                    "raw_entry": entry
                })
    return rows

def print_compliance_summary_table(log_path: str = "compliance_logs.jsonl", truncate_issue: int = 80) -> None:
    """
    Print a compact table of violations with an index column for details.
    """
    rows = flatten_compliance_logs(log_path)
    if not rows:
        print("✅ No violations found (log empty).")
        return

    table = PrettyTable()
    table.field_names = ["Index", "File", "Section", "Issue", "Severity", "Confidence"]
    table.align = "l"
    for idx, r in enumerate(rows):
        issue = r["issue"] or ""
        if len(issue) > truncate_issue:
            issue = issue[: truncate_issue - 3] + "..."
        section = r["section_title"] or "(no title)"
        table.add_row([idx, r["file"], section, issue, r["severity"], f"{r['confidence']:.2f}"])

    print("\n=== Compliance Violations Summary ===")
    print(table)
    print(f"\nTotal Violations: {len(rows)}")
    print("Call show_violation_detail(index) to see full evidence and provenance for a specific violation.\n")

def show_violation_detail(index: int, log_path: str = "compliance_logs.jsonl") -> None:
    """
    Print full details for one flattened flag by index (index from print_compliance_summary_table).
    """
    rows = flatten_compliance_logs(log_path)
    if index < 0 or index >= len(rows):
        print(f"Index {index} out of range (0..{len(rows)-1}).")
        return
    r = rows[index]
    print("\n=== Violation Detail ===")
    print(f"Index: {index}")
    print(f"File: {r['file']}")
    print(f"Section Title: {r['section_title']}")
    print(f"Issue: {r['issue']}")
    print("\nEvidence:")
    print(textwrap.fill(_to_display_text(r['raw_flag'].get("evidence"), maxlen=2000), width=100))
    print("\nRegulatory reference:")
    print(textwrap.fill(_to_display_text(r['raw_flag'].get("reg_ref"), maxlen=2000), width=100))
    print("\nSeverity:", r["severity"])
    print("Confidence:", f"{r['confidence']:.3f}")
    # provenance and structured facts
    raw_entry = r.get("raw_entry") or {}
    print("\nProvenance / context:")
    print("Section metadata:", json.dumps(raw_entry.get("metadata", {}), indent=2))
    print("Structured facts (guideline/reg):")
    print(json.dumps(r.get("raw_entry", {}).get("structured_facts", {}), indent=2))
    print("Timestamp:", raw_entry.get("timestamp"))
    print("\nFull raw flag object:")
    print(json.dumps(r.get("raw_flag", {}), indent=2))

def visualize_reg_graph(graph = None, output_path: str = "reg_graph.png", show = True, seed = 42):
    """
    Visualize the regulations graph for the user for transparency.
    
    Args:
        output_path (str): File path to save the visualization.
    """
    global reg_graph
    if reg_graph is None or reg_graph.number_of_nodes() == 0:
        print("No regulation graph available to visualize.")
        return

    n = reg_graph.number_of_nodes()
    figsize = (max(8, min(24, n * 0.9)), max(6, min(18, n * 0.6)))
    plt.figure(figsize=figsize)
    # tune k based on node count to avoid huge spread
    k = 0.5 if n <= 20 else 1.0 / (n ** 0.5)

    ## change graph layout based on number of nodes. 
    if reg_graph.number_of_nodes() <= 50: ## Current treshold kept at 50
        pos = nx.kamada_kawai_layout(reg_graph)
    else:
        pos = nx.spring_layout(reg_graph, k=0.8, iterations=200, seed=seed)


    degrees = dict(reg_graph.degree())
    node_sizes = [300 + degrees.get(nid, 0) * 200 for nid in reg_graph.nodes()]

    nx.draw_networkx_nodes(reg_graph, pos,
                           node_size=node_sizes,
                           node_color="#bfe9f2",
                           edgecolors="#146b74",
                           linewidths=0.8,
                           alpha=0.95)

    # Draw labels with bbox for readability
    labels = {n: str(n) for n in reg_graph.nodes()}
    nx.draw_networkx_labels(reg_graph, pos, labels, font_size=9, font_weight="bold")

    # Separate edges into self-loop and normal for different drawing styles
    normal_edges = [(u, v) for u, v in reg_graph.edges() if u != v]
    self_edges = [(u, v) for u, v in reg_graph.edges() if u == v]

    nx.draw_networkx_edges(reg_graph, pos, edgelist=normal_edges,
                           arrowstyle='->', arrowsize=12, width=1.0, alpha=0.7)

    # draw self loops as arcs
    if self_edges:
        nx.draw_networkx_edges(reg_graph, pos, edgelist=self_edges,
                               connectionstyle='arc3,rad=0.35',
                               arrowstyle='->', arrowsize=10, width=1.0, alpha=0.9)

    # Edge labels: use 'relation' attribute if present
    edge_labels = {(u, v): data.get("relation", "") for u, v, data in reg_graph.edges(data=True)}
    nx.draw_networkx_edge_labels(reg_graph, pos, edge_labels=edge_labels, font_size=8)

    plt.axis('off')
    # use bbox_inches to avoid tight_layout warnings
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    if show:
       plt.show()
    plt.close()
    print(f"Graph visualization saved at {output_path}")
