# src/compliancechecker.py
from src.utils import check_memory_usage, memory_cleanup
from src.pdf_processing import hybrid_chunking, extract_pdf_text
from src.database import faiss
from src.config import *

def extract_facts_to_json(text: str, source: str = "guideline") -> List[Dict[str, Any]]:
    """
    Robust regex-based fact extraction. Extracts structured facts (quantities, obligations, limits) from free text using regex.
    Prioritizes longer obligation phrases (e.g., 'must not') before shorter ones (e.g., 'must') to avoid conflicting matches.

    Args:
        text (str): Raw guideline/regulation chunk.
        source (str): 'guideline' or 'regulation' (used for tagging).
    
    Returns:
        List[Dict[str, Any]]: Structured facts as JSON-like dicts.
    """
    facts = []

    if not text or not text.strip():
        return facts

    # Pattern for numeric constraints like "85 tomatoes", "200 liters", "5 km"
    num_pattern = re.compile(r'\b(\d+(?:\.\d+)?)\s*([A-Za-z%°μµ/.\-]+)\b')

    # Obligation words (Order Layer Matters) @IMPROVE - Add more obligations
    obligations = { 
        "must not": "prohibited",
        "shall not": "prohibited",
        "may not": "prohibited",
        "must": "mandatory",
        "shall": "mandatory",
        "should": "recommended",
        "may": "permitted"
    }

    # avoid false positives for page/figure markers
    false_units = {"page", "pages", "fig", "figure", "table", "pp"}

    # Scan for numeric constraints
    for match in num_pattern.finditer(text):
        quantity, unit = match.groups()
        unit_norm = unit.lower().strip()
        if unit_norm in false_units:
            continue
        try:
            qty = float(quantity)
        except Exception:
            continue
        facts.append({
            "type": "constraint",
            "source": source,
            "quantity": qty,
            "unit": unit_norm,
            "context": text[max(0, match.start()-40):match.end()+40].strip()
        })

    # Scan for obligations:iterate in order of decreasing phrase length to catch negatives first
    for word, label in sorted(obligations.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE):
            facts.append({
                "type": "obligation",
                "source": source,
                "obligation": label,
                "matched_word": word,
                "context": text[:200]  # snippet
            })

    # dedupe simple (headless) facts by JSON string
    seen = set()
    deduped = []
    for f in facts:
        key = json.dumps(f, sort_keys=True)
        if key not in seen:
            seen.add(key)
            deduped.append(f)
    return deduped

# @COMPLIANCE FUNCTIONS

# Add a new global for stable node ordering / text map
node_id_order = []        # list[str] -> maps FAISS index -> node id
node_id_to_text = {}      # dict[node_id] -> text (for safe lookup)

def extract_entities_relations(markdown: str) -> List[Dict[str, Any]]:
    """
    Robust rule-based extraction of entities (rule IDs, section ids, object mentions)
    and relations from regulation markdown/text.

    Input:
        markdown: full text for a regulation doc (str)

    Output:
        List of triples:
        [
          {
            "head": "Reg101",
            "relation": "requires" | "depends_on" | "overrides" | "applies_to" | "prohibits",
            "tail": "Buffer50m" or "Reg102",
            "evidence": "sentence text containing the relation"
          },
          ...
        ]
    """
    triples = []
    if not markdown or not markdown.strip():
        return triples

    # canonical rule-id regex: supports "Reg 101", "Section 3.4", "Rule12A", etc.
    rule_id_pattern = re.compile(r'\b(?:Reg|Section|Rule)\s*\d+(?:\.\d+)?[A-Z]?\b', flags=re.IGNORECASE)
    rule_ids_found = set([m.group(0).replace(" ", "") for m in rule_id_pattern.finditer(markdown)])

    # Relation keyword groups
    relation_patterns = {
        'requires': ['requires', 'require', 'must', 'shall', 'must comply', 'shall comply', 'required'],
        'depends_on': ['as defined in', 'per', 'see', 'refer to', 'according to'],
        'overrides': ['except', 'notwithstanding', 'supersedes', 'override'],
        'prohibits': ['prohibit', 'forbid', 'must not', 'may not', 'not permitted']
    }

    # lower-case text for quick keyword checks
    lower_text = markdown.lower()

    # Use spaCy for sentence splitting and noun-chunk extraction
    doc = nlp(markdown)
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text:
            continue

        # find rule ids in the sentence (normalized without spaces)
        sent_rule_ids = [m.group(0).replace(" ", "") for m in rule_id_pattern.finditer(sent_text)]

        # find simple named entities (ORG/GPE etc) and noun_chunks as fallback
        ent_texts = [ent.text for ent in sent.ents if ent.text.strip()]
        noun_chunks = [nc.text for nc in sent.noun_chunks if nc.text.strip()]

        # determine relation type by keywords (first match)
        rel_type = 'applies_to'
        st_low = sent_text.lower()
        for name, keywords in relation_patterns.items():
            if any(k in st_low for k in keywords):
                rel_type = name
                break

        # if we have 2+ rule ids, make a direct triple between them
        if len(sent_rule_ids) >= 2:
            head, tail = sent_rule_ids[0], sent_rule_ids[1]
            triples.append({'head': head, 'relation': rel_type, 'tail': tail, 'evidence': sent_text})
            continue

        # if one rule id + other phrase, attach rule -> phrase
        if len(sent_rule_ids) == 1:
            head = sent_rule_ids[0]
            # look for other rule-like ids in sentence (already none), else fallback to a noun_chunk/entity
            tail = None
            if len(ent_texts) >= 1:
                tail = ent_texts[0]
            elif len(noun_chunks) >= 1:
                # pick a short noun chunk that isn't the rule id itself
                for nc in noun_chunks:
                    if nc.replace(" ", "") != head and len(nc) < 120:
                        tail = nc
                        break
            else:
                # fallback: attempt to capture "Buffer 50m" or numeric+unit patterns
                m = re.search(r'\b(buffer|setback|zone)\b[^.]{0,80}', sent_text, flags=re.IGNORECASE)
                tail = m.group(0).strip() if m else sent_text[:120]

            if tail:
                triples.append({'head': head, 'relation': rel_type, 'tail': tail, 'evidence': sent_text})
            continue

        # no explicit rule ids: attempt to pair two entities/noun-chunks
        if len(ent_texts) >= 2:
            head, tail = ent_texts[0], ent_texts[1]
            triples.append({'head': head, 'relation': rel_type, 'tail': tail, 'evidence': sent_text})
            continue

        # fallback: use first two noun chunks
        if len(noun_chunks) >= 2:
            head, tail = noun_chunks[0], noun_chunks[1]
            triples.append({'head': head, 'relation': rel_type, 'tail': tail, 'evidence': sent_text})
            continue

    # deduplicate by head-relation-tail
    unique = {}
    for t in triples:
        key = f"{t['head']}||{t['relation']}||{t['tail']}"
        if key not in unique:
            unique[key] = t

    return list(unique.values())


def build_reg_graph(regs_folder: str = compliance_regulations_folder, rebuild: bool = False) -> None:
    """
    Build or load a regulations graph (NetworkX DiGraph) and FAISS index.

    Inputs:
        regs_folder: path to PDFs folder
        rebuild: force rebuild ignoring cache

    Side effects (globals set):
        reg_graph (nx.DiGraph), faiss_index (faiss index), node_texts (list[str]),
        node_id_order (list[str]), node_id_to_text (dict)
    """
    global reg_graph, faiss_index, node_texts, node_id_order, node_id_to_text

    cache_path = Path(GRAPH_CACHE_JSON)
    emb_path = Path(GRAPH_EMB_CACHE)
    # Try load cache (must include explicit node_id ordering)
    if cache_path.exists() and not rebuild:
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            reg_graph = nx.node_link_graph(data['graph'])
            node_id_order = data.get('node_ids', list(reg_graph.nodes()))
            # node_texts array should correspond to node_id_order
            node_texts = data.get('node_texts', [reg_graph.nodes[n].get('text', '') for n in node_id_order])
           # Load embeddings (prefer .npy, fallback to JSON-embedded list)
            embeddings = None
            if emb_path.exists():
                try:
                    embeddings = np.load(emb_path).astype("float32")
                except Exception as e:
                    logging.warning(f"Failed to load .npy embeddings: {e}")
            else:
                json_emb = data.get("embeddings")
                if json_emb:
                    embeddings = np.asarray(json_emb, dtype="float32")
                    logging.info("Loaded embeddings from JSON fallback.")

            # Verify embedding integrity
            if embeddings is not None:
                embeddings = np.atleast_2d(np.ascontiguousarray(embeddings)).astype("float32")

                if len(node_id_order) != embeddings.shape[0]:
                    logging.warning(
                        f"Node/embedding count mismatch: {len(node_id_order)} nodes vs {embeddings.shape[0]} embeddings. Rebuilding recommended."
                    )

                # Build FAISS index if available
                if "faiss" in globals() and faiss is not None:
                    dim = embeddings.shape[1]
                    faiss_index = faiss.IndexFlatIP(dim)
                    faiss.normalize_L2(embeddings)
                    faiss_index.add(embeddings)
                    logging.info(f"FAISS index loaded (dim={dim}, nodes={len(node_id_order)}).")
                else:
                    faiss_index = None
                    logging.warning("FAISS not available; skipping index creation.")

                node_id_to_text = {
                    nid: node_texts[i] if i < len(node_texts)
                    else reg_graph.nodes[nid].get("text", "")
                    for i, nid in enumerate(node_id_order)
                }

                logging.info("Loaded regulation graph and embeddings from cache.")
                return
            else:
                logging.warning("Embeddings missing in cache, rebuilding...")
        except Exception as e:
            logging.warning(f"Failed to load graph cache ({e}), rebuilding...")

    # Rebuild graph from scratch
    if not os.path.exists(regs_folder):
        logging.error(f"Regulations folder not found: {regs_folder}")
        reg_graph = nx.DiGraph()
        faiss_index = None
        node_texts, node_id_order, node_id_to_text = [], [], {}
        return

    pdf_files = [f for f in os.listdir(regs_folder) if f.lower().endswith(".pdf")]
    all_triples = []

    for file_name in pdf_files:
        pdf_path = os.path.join(regs_folder, file_name)
        try:
            md = extract_pdf_text(pdf_path)
            triples = extract_entities_relations(md)
            for t in triples:
                t.setdefault("source", file_name)
            all_triples.extend(triples)
        except Exception as e:
            logging.warning(f"Failed to parse {file_name}: {e}")

    reg_graph = nx.DiGraph()
    for t in all_triples:
        h, ta = t["head"], t["tail"]
        if "text" not in reg_graph.nodes.get(h, {}):
            reg_graph.add_node(h, text=f"{h}: {t.get('evidence','')[:300]}")
        if "text" not in reg_graph.nodes.get(ta, {}):
            reg_graph.add_node(ta, text=f"{ta}: {t.get('evidence','')[:300]}")
        reg_graph.add_edge(
            h, ta,
            relation=t.get("relation", "applies_to"),
            evidence=t.get("evidence", ""),
            source=t.get("source")
        )

    if reg_graph.number_of_nodes() == 0:
        logging.warning("No nodes found after parsing regulations.")
        reg_graph = nx.DiGraph()
        faiss_index = None
        node_texts, node_id_order, node_id_to_text = [], [], {}
        return

    node_id_order = list(reg_graph.nodes())
    node_texts = [reg_graph.nodes[n].get("text", "") for n in node_id_order]
    node_id_to_text = {nid: node_texts[i] for i, nid in enumerate(node_id_order)}

    try:
        embeddings = np.asarray(embed_model.encode(node_texts, convert_to_numpy=True)).astype("float32")
    except TypeError:
        embeddings = np.asarray(embed_model.encode(node_texts)).astype("float32")

    embeddings = np.atleast_2d(np.ascontiguousarray(embeddings))
    faiss_index = None
    if "faiss" in globals() and faiss is not None:
        dim = embeddings.shape[1]
        faiss_index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings)
        faiss_index.add(embeddings)
        logging.info(f"FAISS index built (dim={dim}, nodes={len(node_id_order)}).")
    else:
        logging.warning("FAISS not available; skipping index creation.")

    # Atomic cache saving in case of crashes
    graph_data = nx.node_link_data(reg_graph)
    cache_data = {
        "graph": graph_data,
        "node_ids": node_id_order,
        "node_texts": node_texts
    }

    # Temporary embedding storage
    tmp_emb = emb_path.with_suffix(".tmp.npy")    # e.g. reg_graph_embeddings.tmp.npy
    tmp_json = cache_path.with_suffix(".tmp")     # e.g. reg_graph.json.tmp

    try:
        np.save(tmp_emb, embeddings)
        os.replace(tmp_emb, emb_path)

        # Save JSON metadata to temporary file and move into place
        with open(tmp_json, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2)
        os.replace(tmp_json, cache_path)
        logging.info(f"Built and cached regulation graph ({len(node_id_order)} nodes, {reg_graph.number_of_edges()} edges).")
    except Exception as e:
        logging.warning(f"Failed to write cache files: {e}")

def graph_enhanced_retrieval(section_text: str, top_k = TOP_K_DEFAULT, traversal_depth: int = 1) -> str:
    """
    Hybrid retrieval: embed the query, search FAISS, map results to node IDs using node_id_order,
    and return a short linearized subgraph that includes relations/evidence.
    Returns a concise string (trimmed) suitable to pass to the LLM prompt.
    """
    global reg_graph, faiss_index, node_texts, node_id_order, node_id_to_text

    if reg_graph is None or faiss_index is None or not node_id_order:
        build_reg_graph()

    if reg_graph is None or faiss_index is None or not node_id_order:
        return "No regulations available."

    # embed query robustly
    try:
        q_emb = np.asarray(embed_model.encode([section_text], convert_to_numpy=True)).astype('float32')
    except TypeError:
        q_emb = np.asarray(embed_model.encode([section_text])).astype('float32')

    q_emb = np.ascontiguousarray(q_emb)
    faiss.normalize_L2(q_emb)
    try:
        scores, indices = faiss_index.search(q_emb, top_k)
    except Exception as e:
        logging.error(f"FAISS search failed: {e}")
        return "Regulations retrieval failed."

    retrieved_parts = []
    for i, idx in enumerate(indices[0]):
        if idx == -1 or idx >= len(node_id_order):
            continue
        node_id = node_id_order[idx]
        node_text = node_id_to_text.get(node_id, reg_graph.nodes[node_id].get('text', ''))
        # collect neighbors (ego graph) due to traversal_depth
        subgraph = nx.ego_graph(reg_graph, node_id, radius=traversal_depth)
        # linearize: node + immediate edges (limit lengths)
        part = f"[{node_id}] {node_text[:300]}"
        for u, v, data in subgraph.edges(data=True):
            rel = data.get('relation', '')
            evidence = data.get('evidence', '')[:180]
            part += f" -> {u} [{rel}] {v}: {evidence}..."
        retrieved_parts.append(part[:1200])  # trim each retrieved part to avoid huge prompts

    if not retrieved_parts:
        return "No relevant regulations found."

    return "Relevant Regulations with Relations: " + " || ".join(retrieved_parts)


def _extract_json_array_from_text(text: str) -> Optional[str]:
    """
    Helper: extract the first JSON array substring (a [...] block) from text (dotall).
    Returns the substring or None if not found.
    """
    m = re.search(r'(\[[\s\S]*\])', text)
    if m:
        return m.group(1)
    return None

def split_pdf_into_sections(pdf_path: str) -> List[Dict[str, Any]]:
    """Split project PDF into semantic sections using hybrid_chunking.

    Args:
        pdf_path (str): Path to project PDF.

    Returns:
        List[Dict[str, Any]]: Sections as [{'title': str, 'content': str, 'metadata': dict}].
    """
    text = extract_pdf_text(pdf_path)
    chunks = hybrid_chunking(text)
    sections = []
    for i, chunk in enumerate(chunks):
        title = chunk['headers'][0] if chunk['headers'] else f"Section {i+1}"
        metadata = {"estimated_tokens": len(chunk['text'].split()), "source": pdf_path}
        sections.append({"title": title, "content": chunk['text'], "metadata": metadata})
    return sections

def check_compliance_for_section_with_regex(section: Dict[str, Any], rag_regs: Optional[str], model_name: str = str(LLM_MODEL_NAME)) -> Dict[str, Any]:
    """
    Enhanced compliance check:
    - Extracts structured JSON facts from section and regs using regex.
    - Passes both raw text and extracted JSON to AI.
    - Saves JSON, query, and AI answer to a log file for transparency.
    - Extracts structured JSON facts from section and regs using regex.
    - If rag_regs is not provided, compute it via graph_enhanced_retrieval.
    - Calls the LLM and attempts robust JSON extraction.

    Returns:
        dict with 'flags' (AI output), 'facts' (structured facts used), 'answer' (LLM raw answer)
    """
    section_text = section.get("content", "") or ""
    if rag_regs is None:
        try:
            rag_regs = graph_enhanced_retrieval(section_text)
        except Exception as e:
            rag_regs = ""

    # Extract structured facts
    guideline_facts = extract_facts_to_json(section_text, source="guideline")
    reg_facts = extract_facts_to_json(rag_regs or "", source="regulation")
    structured_json = {"guideline_facts": guideline_facts, "regulation_facts": reg_facts}

    # Compliance prompt for AI
    prompt = f"""
You are a regulatory compliance expert.
Here is a project section and relevant regulations.

Section Title: {section.get('title')}
Section Content: {section_text}

Relevant Regulations: {rag_regs}

Here are structured facts extracted with regex:
{json.dumps(structured_json, indent=2)}

Use BOTH the raw text and structured facts to check for violations.
Output PURE JSON list of flags with keys:
["issue","evidence","reg_ref","severity","confidence"].
"""

    output = ""   # always defined
    flags: List[Dict[str, Any]] = []

    # Call LLM
    try:
        response = llm.chat(model=model_name, messages=[{'role': 'user', 'content': prompt}],
                           options={"temperature": 0.4, "top_p": 0.9})
        # extract text robustly
        if isinstance(response, dict) and 'message' in response and isinstance(response['message'], dict):
            output = response['message'].get('content', '')
        elif hasattr(response, 'message') and hasattr(response.message, 'content'):
            output = response.message.content
        else:
            output = str(response)
        output = output.strip()

        # Try direct parse, then fallback to JSON substring extraction
        try:
            parsed = json.loads(output)
            if isinstance(parsed, list):
                flags = parsed
        except Exception:
            json_sub = _extract_json_array_from_text(output)
            if json_sub:
                try:
                    parsed = json.loads(json_sub)
                    if isinstance(parsed, list):
                        flags = parsed
                except Exception:
                    flags = []
            else:
                flags = []
    except Exception as e:
        logging.error(f"Ollama connection failed: {e}")
        print("\n[ERROR] Ollama is not running. Please start it using:\n   ollama serve\n")
        return {"flags": [], "facts": {}, "answer": "Ollama not available"}

    # normalize flags: ensure list of dicts, coerce textual fields to strings, normalize confidence values
    safe_flags = []
    for f in flags:
        if not isinstance(f, dict):
            continue

        # Normalize textual fields to strings (handles lists, None, numbers)
        issue = f.get("issue", "") or ""
        if isinstance(issue, (list, tuple)):
            issue = " ".join(map(str, issue))
        issue = str(issue)
        f["issue"] = issue

        evidence = f.get("evidence", "") or ""
        if isinstance(evidence, (list, tuple)):
            evidence = " ".join(map(str, evidence))
        evidence = str(evidence)
        f["evidence"] = evidence

        reg_ref = f.get("reg_ref", "") or ""
        if isinstance(reg_ref, (list, tuple)):
            reg_ref = " ".join(map(str, reg_ref))
        reg_ref = str(reg_ref)
        f["reg_ref"] = reg_ref

        severity = f.get("severity", "medium") or "medium"
        if not isinstance(severity, str):
            severity = str(severity)
        f["severity"] = severity
        
        # normalize confidence numeric and clamp 0..1
        try:
            conf = float(f.get('confidence', 0.5))
        except Exception:
            conf = 0.5
        conf = max(0.0, min(1.0, conf))
        f['confidence'] = conf

        safe_flags.append(f)

    # enrich each flag with confidence scoring
    safe_flags = [assign_confidence(f, guideline_facts, reg_facts) for f in safe_flags]

    # Save transparency log
    log_entry = {
        "section_title": section.get("title"),
        "section_content": section_text[:400],
        "relevant_regs": rag_regs[:400],
        "structured_facts": structured_json,
        "ai_answer": output,
        "flags": safe_flags,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
    }
    
    try:
        with open("compliance_logs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        logging.warning(f"Failed to write compliance log: {e}")

    return {"flags": safe_flags, "facts": structured_json, "answer": output}

def check_file_compliance(project_desc: str, project_pdf_path: str) -> Dict[str, Any]:
    """
    Compliance checker: Splits project PDF, checks sections aginst regulations graph RAG, aggregates flags.

    Returns a report dict that preserves the original keys and includes:
      - section_reports: List[ { index, title, flags, error (or None), duration_s, skipped } ]
      - processing_time_s: total time spent for this file
    """
    start_total = time.time()
    project_pdf_path = os.path.abspath(project_pdf_path)

    if not os.path.exists(project_pdf_path):
        raise ValueError(f"Project PDF not found: {project_pdf_path}")

    # Attempt to split into sections; propagate error (caller typically handles it)
    try:
        sections = split_pdf_into_sections(project_pdf_path)
    except Exception as e:
        logging.exception(f"Failed to split PDF into sections: {project_pdf_path}")
        # Keep the raising behavior for a missing/invalid PDF as before.
        raise

    all_flags: List[Dict[str, Any]] = []
    section_reports: List[Dict[str, Any]] = []

    if not sections:
        # No content extracted — return an explicit report (keeps original keys)
        report = {
            "project_desc": project_desc,
            "project_pdf": project_pdf_path,
            "sections_checked": 0,
            "section_reports": [],
            "flags": [],
            "overall_summary": {
                "total_flags": 0,
                "compliance_status": "No content extracted"
            },
            "processing_time_s": time.time() - start_total
        }
        return report

    # Process each section independently; capture exceptions per-section
    for idx, section in enumerate(sections):
        t0 = time.time()
        sec_report: Dict[str, Any] = {
            "index": idx,
            "title": section.get("title"),
            "metadata": section.get("metadata", {}),
            "flags": [],
            "error": None,
            "duration_s": None,
            "skipped": False
        }

        # Lightweight skip: skip sections with essentially no content
        content = (section.get("content") or "").strip()
        if len(content) < 20:
            sec_report["skipped"] = True
            sec_report["duration_s"] = 0.0
            section_reports.append(sec_report)
            continue

        try:
            # compute RAG regs for this section and call the checker
            rag_regs = graph_enhanced_retrieval(content)
            result = check_compliance_for_section_with_regex(section, rag_regs=rag_regs)
            # result should be dict; get flags safely
            if isinstance(result, dict):
                flags = result.get("flags", [])
            elif isinstance(result, list):
                flags = result
            else:
                flags = []

            if not isinstance(flags, list):
                logging.warning(f"Unexpected flags type for section {idx} in {project_pdf_path}")
                flags = []

            # annotate each flag with section index/title if not already present
            for f in flags:
                if isinstance(f, dict):
                    f.setdefault("section_index", idx)
                    f.setdefault("section_title", section.get("title"))
            sec_report["flags"] = flags
            all_flags.extend(flags)
        except Exception as e:
            logging.exception(f"Error checking compliance for section {idx} in {project_pdf_path}: {e}")
            sec_report["error"] = str(e)
            # Continue processing other sections (do not re-raise)
        finally:
            sec_report["duration_s"] = time.time() - t0
            section_reports.append(sec_report)

        # Optional: memory check after each section (uncomment if desired)
        if not check_memory_usage():
            memory_cleanup()

    total_flags = len(all_flags)
    processing_time = time.time() - start_total

    report = {
        "project_desc": project_desc,
        "project_pdf": project_pdf_path,
        "sections_checked": len(sections),
        "section_reports": section_reports,
        "flags": all_flags,
        "overall_summary": {
            "total_flags": total_flags,
            "compliance_status": "Compliant" if total_flags == 0 else "Non-Compliant (Review Required)"
        },
        "processing_time_s": processing_time
    }
    return report


def check_project_compliance(compliance_folder = project_guidelines_folder, project_desc: str = None):
    """
    Batch compliance checker: Processes all PDFs in the project guidelines folder folder, then generates individual compliance reports .
    """
    start_batch = time.time()
    if not os.path.exists(compliance_folder):
        logging.error(f"Compliance folder not found: {compliance_folder}")
        return {"error": f"Folder not found: {compliance_folder}"}

    # deterministic order and case-insensitive pdf detection
    pdf_files = sorted([f for f in os.listdir(compliance_folder) if f.lower().endswith(".pdf")])

    if not pdf_files:
        logging.warning(f"No PDF files found in {compliance_folder}")
        return {"error": "No PDF files found in folder"}

    logging.info(f"Processing {len(pdf_files)} PDFs for compliance checking...")
    print(f"\nProcessing {len(pdf_files)} project PDFs for compliance...\n")

    all_reports: List[Dict[str, Any]] = []
    total_flags = 0
    files_checked = 0
    files_failed = 0

    for pdf_file in tqdm(pdf_files, desc="Compliance Progress", unit="file", ncols=80): ## Create progress bar for entire process. 
        pdf_path = os.path.join(compliance_folder, pdf_file)
        desc = project_desc or f"Compliance check for {pdf_file}" ##IMPROVE##
        start_time = time.time() ## Time overall process
        print(f"Checking: {pdf_file}...")
        file_start = time.time() ## Time individual files in log

        try:
            report = check_file_compliance(desc, pdf_path)
            duration = time.time() - start_time
            report["processing_time_sec"] = round(duration, 2)
            # Ensure a dict
            if not isinstance(report, dict):
                raise RuntimeError("Child report is not a dict")

            # attach filename and timing
            report["filename"] = pdf_file
            report["processing_time_s"] = report.get("processing_time_s", time.time() - file_start)

            all_reports.append(report)
            # safe read of flags from child report
            file_flags = report.get("overall_summary", {}).get("total_flags", 0)
            try:
                file_flags = int(file_flags)
            except Exception:
                file_flags = 0
            total_flags += file_flags
            files_checked += 1

            status = "✓ Compliant" if file_flags == 0 else f"✗ {file_flags} violations"
            print(f"  {status}\n")

        except Exception as e:
            logging.exception(f"Failed to process {pdf_file}: {e}")
            files_failed += 1
            print(f"  ✗ Error processing file: {e}\n")

    batch_time = time.time() - start_batch
    aggregated_report = {
        "batch_description": project_desc or "Batch compliance check",
        "compliance_folder": compliance_folder,
        "files_checked": files_checked,
        "files_failed": files_failed,
        "total_pdfs": len(pdf_files),
        "total_violations": total_flags,
        "individual_reports": all_reports,
        "overall_summary": {
            "compliant_files": sum(1 for r in all_reports if r.get("overall_summary", {}).get("total_flags", 0) == 0),
            "non_compliant_files": sum(1 for r in all_reports if r.get("overall_summary", {}).get("total_flags", 0) > 0),
            "total_flags_across_all_files": total_flags,
            "batch_status": "All Compliant" if total_flags == 0 else f"{total_flags} Total Violations Across {files_checked} Files"
        },
        "processing_time_s": batch_time
    }
    return aggregated_report

def assign_confidence(flag: Dict[str, Any], guideline_facts: List[Dict], reg_facts: List[Dict], retrieval_score: float = None)-> Dict[str, Any]:
    """
    Assigns a confidence score to a compliance flag using deterministic + heuristic rules.
    Deterministic matches override AI output.
    """
    conf = 0.5
    label = "Medium"
    breakdown = {}

    # 1. Deterministic checks (regex fact match)
    issue_text = str(flag.get("issue", "") or "").lower()
    evidence_text = str(flag.get("evidence", "") or "").lower()

    det_match = False
    for fact in (guideline_facts or []) + (reg_facts or []):
        if fact.get("type") == "constraint":
            q = fact.get("quantity")
            u = str(fact.get("unit", "") or "").lower()
            if q is None:
                continue
            q_str = str(q)
            if q_str and q_str in evidence_text and (u == "" or u in evidence_text):
                det_match = True
                break

    if det_match:
        conf = 0.95
        label = "Very High"
        breakdown["deterministic_match"] = True
    else:
        breakdown["deterministic_match"] = False

    # Use AI confidence if present
    ai_conf = flag.get("confidence")
    if isinstance(ai_conf, (int, float)):
        breakdown["ai_confidence"] = ai_conf
        conf = max(conf, float(ai_conf))

    # Retrieval quality (if provided)
    if retrieval_score is not None:
        breakdown["retrieval_score"] = retrieval_score
        if retrieval_score > 0.8:
            conf = max(conf, 0.85)
        elif retrieval_score < 0.4:
            conf = min(conf, 0.6)

    # Map confidence score to a label
    if conf >= 0.9: label = "Very High"
    elif conf >= 0.75: label = "High"
    elif conf >= 0.5: label = "Medium"
    elif conf >= 0.25: label = "Low"
    else: label = "Very Low"

    flag["confidence"] = round(conf, 3)
    flag["confidence_label"] = label
    flag["confidence_breakdown"] = breakdown
    return flag