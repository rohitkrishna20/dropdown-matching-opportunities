from flask import Flask, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
import json, re, os, requests, ollama

app = Flask(__name__)

FEEDBACK_PATH = "feedback_memory.json"
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# -------------------- Feedback store --------------------
def load_feedback():
    if os.path.exists(FEEDBACK_PATH):
        try:
            with open(FEEDBACK_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    data.setdefault("correct", {})
                    data.setdefault("incorrect", {})
                    data.setdefault("last_run", {})
                    return data
        except Exception as e:
            print(f"⚠️ load_feedback: {e}")
    return {"correct": {}, "incorrect": {}, "last_run": {}}

def save_feedback():
    try:
        with open(FEEDBACK_PATH, "w", encoding="utf-8") as f:
            json.dump(feedback_memory, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ save_feedback: {e}")

feedback_memory = load_feedback()

# -------------------- Text utils --------------------
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower()) if isinstance(s, str) else s

_UI_STOP = {"dashboard", "page", "view", "panel", "tile", "card", "module", "section", "tab"}

def _tokens_core(s: str) -> list[str]:
    if not isinstance(s, str): 
        return []
    s = re.sub(r"(?<!^)(?=[A-Z])", " ", s)     # camelCase -> camel Case
    s = re.sub(r"[^a-zA-Z0-9]+", " ", s)       # keep alnum separators
    return [t for t in s.strip().lower().split() if t]

def _tokens(s: str) -> list[str]:
    toks = _tokens_core(s)
    return [t for t in toks if t not in _UI_STOP]

def _jaccard(a: set, b: set) -> float:
    if not a and not b: return 0.0
    return len(a & b) / max(1, len(a | b))

def _is_mostly_upper(s: str) -> bool:
    letters = [c for c in s if c.isalpha()]
    if not letters: return False
    upp = sum(c.isupper() for c in letters)
    return upp / len(letters) >= 0.8 and len(letters) >= 4

def _is_mostly_numeric(s: str) -> bool:
    digits = sum(c.isdigit() for c in s)
    return digits / max(1, len(s)) >= 0.4

ID_BASE62 = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-")

def _looks_like_hex_id(s: str) -> bool:
    w = s.strip()
    return bool(re.fullmatch(r"[0-9a-fA-F]{10,}", w))

def _looks_like_base62_id(s: str) -> bool:
    w = s.strip()
    return len(w) >= 10 and " " not in w and all(c in ID_BASE62 for c in w)

def _looks_like_id(s: str) -> bool:
    if not isinstance(s, str):
        return False
    t = s.strip()
    if not t: return False
    if " " not in t and len(t) >= 8:
        alnum_ratio = sum(c.isalnum() for c in t) / len(t)
        if alnum_ratio >= 0.9 and any(c.isdigit() for c in t):
            return True
    if _looks_like_hex_id(t) or _looks_like_base62_id(t):
        return True
    if re.fullmatch(r"[A-Za-z0-9_-]{12,}", t):
        return True
    return False

def _is_headerish(s: str) -> bool:
    if not isinstance(s, str): return False
    w = s.strip()
    if not w: return False
    if _looks_like_id(w): return False
    if any(ch.isdigit() for ch in w): return False
    if "_" in w or "#" in w: return False
    parts = w.split()
    if not (1 <= len(parts) <= 3): return False
    if any(len(p) > 28 for p in parts): return False
    if _is_mostly_upper(s): return False
    if _is_mostly_numeric(s): return False
    if not any(c.isalpha() for c in s): return False
    return True

def _valid_figma_label(t: str) -> bool:
    if not isinstance(t, str): return False
    s = t.strip()
    if not s: return False
    # drop obvious IDs / codes / refs
    if _looks_like_hex_id(s) or _looks_like_base62_id(s): return False
    # shape constraints for column headers
    if "_" in s or "#" in s: return False
    # allow hyphens (Figma titles often have en/em dashes)
    # if "-" in s: return False   # ← removed
    if len(s) > 30: return False
    if len(s.split()) > 4: return False
    return _is_headerish(s)


# -------------------- Feedback blocklist --------------------
def build_blocklist() -> set:
    blocked = set()
    inc = feedback_memory.get("incorrect", {}) or {}
    for hdr, pats in inc.items():
        blocked.add(_norm(hdr))
        if isinstance(pats, list):
            for p in pats:
                if isinstance(p, str):
                    blocked.add(_norm(p))
    return blocked

# -------------------- JSON helpers --------------------
def force_decode(raw):
    try:
        rounds = 0
        while isinstance(raw, str) and rounds < 6:
            rounds += 1
            try:
                raw = json.loads(raw)
                continue
            except Exception:
                break
        return raw
    except Exception:
        return raw

def get_payload(req):
    payload = req.get_json(silent=True)
    if isinstance(payload, dict) and payload:
        return payload
    return None

def _root_token(h: str) -> str:
    t = _tokens(h)
    return t[0] if t else ""


# -------------------- Figma harvest --------------------
def extract_figma_text(figma_json: dict) -> list[str]:
    """
    Preferred: TEXT nodes' 'characters'.
    Fallback (same pass): any 'name' values (Frame/Group/Component) -> split camel/kebab/_ and clean.
    """
    out = []

    def is_numeric(t: str) -> bool:
        cleaned = t.replace(",", "").replace("%", "").replace("$", "").strip()
        return cleaned.replace(".", "").isdigit()

    def add_clean(s: str):
        if not isinstance(s, str): return
        s2 = s.strip()
        if not s2: return
        # convert camel/kebab/underscores to spaces, titlecase where appropriate
        s_norm = " ".join(_tokens_core(s2)).strip()
        s_norm = re.sub(r"\s+", " ", s_norm)
        s_title = " ".join(w.capitalize() for w in s_norm.split())
        # keep both original (if looks like plain text) and the cleaned title
        cands = [s2, s_title]
        for c in cands:
            if c and not is_numeric(c) and _is_headerish(c):
                out.append(c)

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "TEXT":
                val = node.get("characters", "")
                if isinstance(val, str):
                    add_clean(val)
            # also allow 'name' fields because some dashboard files lack TEXT nodes
            nm = node.get("name")
            if isinstance(nm, str):
                add_clean(nm)
            for v in node.values():
                if isinstance(v, (dict, list)):
                    walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(figma_json)

    # de-dup preserve order + apply validator
    seen, uniq = set(), []
    for s in out:
        if _valid_figma_label(s):
            if s not in seen:
                seen.add(s); uniq.append(s)
    return uniq

# -------------------- RHS field universe --------------------
def extract_all_keys(data, prefix=""):
    keys = set()
    if isinstance(data, dict):
        for k, v in data.items():
            full = f"{prefix}.{k}" if prefix else k
            keys.add(full)
            keys.update(extract_all_keys(v, full))
    elif isinstance(data, list):
        for item in data:
            keys.update(extract_all_keys(item, prefix))
    return keys

def field_leaf(path: str) -> str:
    """Prefer token after last '.properties.'; else last dotted token; strip x- extensions."""
    if not isinstance(path, str) or not path: return ""
    parts = path.split(".")
    try:
        last_prop = max(i for i, t in enumerate(parts) if t == "properties")
        j = last_prop + 1
        if j < len(parts):
            leaf = parts[j]
            if leaf == "items" and j + 1 < len(parts):
                leaf = parts[j + 1]
        else:
            leaf = parts[-1]
    except ValueError:
        leaf = parts[-1]
    if leaf.startswith("x-"):
        for tok in reversed(parts):
            if not tok.startswith("x-"):
                leaf = tok
                break
    return leaf

# ---- schema aliasing: turn long OpenAPI-ish ids into human words ----
def _id_tokens(s: str):
    s1 = re.sub(r"(?<!^)(?=[A-Z])", " ", s)
    return [t for t in re.split(r"[^A-Za-z0-9]+", s1) if t]

def _schema_alias(s: str) -> str:
    toks = _id_tokens(s)
    drop = {"data","dy","homepage","home","mgr","vbc","get","one","all","put","post",
            "del","delete","response","resp","bo","id","list","key","schema","schemas","components"}
    core = [t for t in toks if t.lower() not in drop]
    low = [t.lower() for t in toks]
    if "sales" in low and "dashboard" in low:
        return "Sales Dashboard"
    if core:
        head = [core[0].capitalize()] + [c.lower() for c in core[1:3]]
        return " ".join(head)
    return s

def collect_rhs_paths_and_leaves(data):
    paths = sorted(list(extract_all_keys(data)))
    meta = []
    for p in paths:
        leaf = field_leaf(p)
        if not leaf: 
            continue
        meta.append({"path": p, "leaf": leaf})
        alias = _schema_alias(leaf)
        if alias and alias != leaf:
            meta.append({"path": p, "leaf": alias, "alias_of": leaf})
    return meta

def build_faiss_on_leaves(rhs_meta):
    docs = []
    for m in rhs_meta:
        docs.append(Document(page_content=m["leaf"], metadata={"path": m["path"]}))
    return FAISS.from_documents(docs, OllamaEmbeddings(model=OLLAMA_MODEL))

# -------------------- LLM prompt --------------------
def make_prompt_from_figma(labels: list[str]) -> str:
    blob = "\n".join(f"- {t}" for t in labels)
    incorrect = set(p for pats in feedback_memory["incorrect"].values() for p in pats)
    correct   = set(p for pats in feedback_memory["correct"].values()   for p in pats)
    avoid = ("\nAvoid patterns like:\n" + "\n".join(f"- {p}" for p in incorrect)) if incorrect else ""
    prefer = ("\nPrefer patterns like:\n" + "\n".join(f"- {p}" for p in correct)) if correct else ""

    return f"""
Extract TABLE COLUMN HEADERS from the candidate label list.

Rules (follow ALL strictly):
- Use ONLY labels that appear in the list (do not invent).
- Keep them short (1–3 words), human-readable, column-like (not actions/menus/status).
- ALWAYS RETURN an output - never have any empty headers!
- DO NOT select labels that contain an underscore "_" or a hash "#".
- Avoid generic technical/container words such as: components, schemas, properties, paths, tags, servers, definitions, refs.

{avoid}
{prefer}

Return STRICT JSON: keys = your normalized headers, values = the EXACT matched label.

Candidate labels:
{blob}
""".strip()

# -------------------- Matching helpers --------------------
def _header_aliases(h: str) -> list[str]:
    hn = _norm(h)
    aliases = [hn]
    map_simple = {
        "at risk": "risk",
        "risk status": "risk",
        "created on": "created",
        "created at": "created",
        "source type": "source",
        "primary flag": "primary",
        "account name": "account",
        "company": "account",
    }
    if hn in map_simple:
        aliases.append(map_simple[hn])
    ht = _tokens_core(hn)
    if len(ht) == 2 and ht[1] in _UI_STOP:
        aliases.append(ht[0])
    return list(dict.fromkeys(aliases))

def has_rhs_affinity(header: str, rhs_meta: list[dict], min_overlap: float = 0.30) -> bool:
    if not isinstance(header, str) or not header.strip():
        return False
    candidates = _header_aliases(header)

    for cand in candidates:
        htoks = set(_tokens(cand))
        hnorm = _norm(cand)
        for m in rhs_meta:
            leaf = m.get("leaf") or ""
            if not leaf: continue
            ln = _norm(leaf)
            ltoks = set(_tokens(leaf))

            if ln == hnorm:
                return True
            if hnorm and (hnorm in ln or ln in hnorm):
                return True
            if htoks and ltoks and _jaccard(htoks, ltoks) >= min_overlap:
                return True
    return False

def rank_candidates_for(header: str, rhs_meta: list[dict], field_index, k: int = 3):
    variants = _header_aliases(header)
    scored = []

    def score_pair(hcand: str, leaf: str):
        hnorm = _norm(hcand)
        htoks = set(_tokens(hcand))
        ln = _norm(leaf)
        ltoks = set(_tokens(leaf))
        if ln == hnorm: return 0.0
        if hnorm and (hnorm in ln or ln in hnorm): return 0.25
        j = _jaccard(htoks, ltoks)
        if j > 0: return 1.0 - min(0.99, j)
        return None

    for m in rhs_meta:
        leaf = m.get("leaf") or ""
        best = None
        for hc in variants:
            s = score_pair(hc, leaf)
            if s is not None:
                best = s if best is None else min(best, s)
        if best is not None:
            # tiny preference for alias leaves (human readable)
            if "alias_of" in m:
                best = max(0.0, best - 0.05)  # lower score = better
            scored.append((best, m["path"], leaf))

    scored.sort(key=lambda x: x[0])
    out = [{"field": p, "field_short": leaf, "score": float(s)} for (s, p, leaf) in scored[:k]]

    if len(out) < k:
        try:
            acc = []
            for hc in variants:
                acc.extend(field_index.similarity_search_with_score(hc, k=k*2))
            seen_paths = {x["field"] for x in out}
            for doc, dist in acc:
                pth = doc.metadata.get("path", "")
                leaf = doc.page_content
                if pth in seen_paths:
                    continue
                out.append({"field": pth, "field_short": leaf, "score": float(dist) + 0.5})
                seen_paths.add(pth)
                if len(out) >= k:
                    break
        except Exception:
            pass

    out.sort(key=lambda x: x.get("score", 1e9))
    return out[:k]

# -------------------- Ollama info --------------------
def get_ollama_info():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if r.status_code == 200:
            return {"ollama_host": OLLAMA_URL, "models": [m.get("name") for m in r.json().get("models", [])]}
        return {"ollama_host": OLLAMA_URL, "error": f"status {r.status_code}"}
    except Exception as e:
        return {"ollama_host": OLLAMA_URL, "error": str(e)}

@app.get("/api/ollama_info")
def api_ollama_info():
    return jsonify(get_ollama_info())

@app.post("/api/find_fields")
def api_find_fields():
    try:
        raw = get_payload(request)
        if not isinstance(raw, dict):
            return jsonify({"error": "Request must include figma_json and data_json"}), 400
        if "figma_json" not in raw or "data_json" not in raw:
            return jsonify({"error": "Missing 'figma_json' or 'data_json' keys"}), 400

        figma_json = force_decode(raw["figma_json"])
        data_json  = force_decode(raw["data_json"])

        # RHS universe + FAISS index (readable matching only; no effect on header set)
        rhs_meta = collect_rhs_paths_and_leaves(data_json)
        field_index = build_faiss_on_leaves(rhs_meta)

        # ---------- Figma harvest ONLY (TEXT.characters + node.name, cleaned) ----------
        figma_labels = extract_figma_text(figma_json)

        # Emergency sweep (still Figma-only: harvest strings inside Figma JSON, not RHS)
        if not figma_labels:
            def _harvest_strings(node, bag):
                if isinstance(node, dict):
                    for v in node.values():
                        if isinstance(v, str):
                            s = v.strip()
                            if 1 <= len(s) <= 40 and _is_headerish(s):
                                bag.append(s)
                        elif isinstance(v, (dict, list)):
                            _harvest_strings(v, bag)
                elif isinstance(node, list):
                    for it in node:
                        _harvest_strings(it, bag)

            _tmp = []
            _harvest_strings(figma_json, _tmp)
            # basic hygiene
            _BAD_TERMS = {"components", "schemas", "properties", "responses", "schema", "paths", "tags", "servers", "definitions", "refs"}
            seen = set()
            figma_labels = []
            for s in _tmp:
                if s in seen: 
                    continue
                seen.add(s)
                if "_" in s or "#" in s: 
                    continue
                if _norm(s) in _BAD_TERMS: 
                    continue
                if _valid_figma_label(s):
                    figma_labels.append(s)

        # Final candidate pool: STRICTLY from Figma, validated, de-duplicated (exact by _norm)
        cand_pool = []
        seen_norm = set()
        for lbl in figma_labels:
            if _valid_figma_label(lbl):
                n = _norm(lbl)
                if n not in seen_norm:
                    seen_norm.add(n)
                    cand_pool.append(lbl)

        # If Figma truly has nothing usable, return empty headers (strict guarantee)
        if not cand_pool:
            headers = []
            matches = {}
            return jsonify({"headers_extracted": headers, "matches": matches})

        # ---------- Optional LLM pass (still Figma-only selection) ----------
        headers = []
        try:
            prompt = make_prompt_from_figma(cand_pool)
            out = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}])
            content = (out.get("message") or {}).get("content", "") or ""
            parsed = {}
            if content:
                try:
                    parsed = json.loads(content) or {}
                except Exception:
                    m = re.search(r"\{[\s\S]*?\}", content)
                    if m:
                        try:
                            parsed = json.loads(m.group())
                        except Exception:
                            parsed = {}
            if isinstance(parsed, dict):
                norm_fig = {_norm(x): x for x in cand_pool}
                for _, v in parsed.items():
                    if isinstance(v, str):
                        nv = _norm(v)
                        if nv in norm_fig:
                            headers.append(norm_fig[nv])
        except Exception:
            headers = []

        # ---------- Fallback: purely heuristic ranking (still Figma-only) ----------
        def _shape_score(s: str) -> float:
            parts = s.strip().split()
            L = sum(len(p) for p in parts)
            alpha = sum(c.isalpha() for c in s)
            shout = 1.0 if _is_mostly_upper(s) else 0.0
            # lower is better
            return L - 2.0*shout - 0.5*abs(len(parts)-2) - 0.25*alpha

        if not headers:
            headers = sorted(
                cand_pool,
                key=lambda x: (0 if has_rhs_affinity(x, rhs_meta, min_overlap=0.20) else 1,
                               _shape_score(x))
            )

        # Belt & suspenders: ensure headers are a subset of Figma labels only
        fig_set_norm = {_norm(x) for x in figma_labels}
        headers = [h for h in headers if _norm(h) in fig_set_norm]

        # Cap size (your existing max)
        headers = headers[:15]

        # ---------- Matches (RHS-only, read-only; does not affect headers) ----------
        matches = {}
        for h in headers:
            matches[h] = rank_candidates_for(h, rhs_meta, field_index, k=3)

        # Save feedback context (safe: uses headers as chosen above)
        feedback_memory["last_run"] = {}
        for h in headers:
            feedback_memory["correct"].setdefault(h, []).append(h)
            feedback_memory["last_run"][h] = {
                "matched_ui_label": h,
                "figma_text": figma_labels,
                "top_rhs_candidates": matches.get(h, [])
            }
        save_feedback()

        # Runtime assertion: guarantee headers came from Figma
        assert all(_norm(h) in fig_set_norm for h in headers), \
            "Internal error: header not from Figma labels"

        if request.args.get("debug") in {"1","true"}:
            return jsonify({
                "headers_extracted": headers,
                "matches": matches,
                "debug": {
                    "figma_label_count": len(figma_labels),
                    "figma_sample": figma_labels[:25],
                    "rhs_paths_count": len(rhs_meta),
                    "rhs_leaf_sample": [m["leaf"] for m in rhs_meta[:25]]
                }
            })

        return jsonify({"headers_extracted": headers, "matches": matches})

    except Exception as e:
        return jsonify({"error": "Find fields failed", "details": str(e)}), 500

# -------------------- API: feedback --------------------
@app.post("/api/feedback")
def api_feedback():
    try:
        body = request.get_json(force=True)
        header = body.get("header")
        status = body.get("status")  # "correct" | "incorrect"
        if not header or status not in {"correct", "incorrect"}:
            return jsonify({"error": "Invalid feedback format"}), 400

        patterns = feedback_memory["correct"].get(header, [])

        if status == "correct":
            feedback_memory["correct"].setdefault(header, [])
            for p in patterns:
                if p not in feedback_memory["correct"][header]:
                    feedback_memory["correct"][header].append(p)
        else:
            feedback_memory["incorrect"].setdefault(header, [])
            if header not in feedback_memory["incorrect"][header]:
                feedback_memory["incorrect"][header].append(header)
            for p in patterns:
                if p not in feedback_memory["incorrect"][header]:
                    feedback_memory["incorrect"][header].append(p)
            feedback_memory["correct"].pop(header, None)

        ctx = (feedback_memory.get("last_run") or {}).get(header, {})
        matched_ui = ctx.get("matched_ui_label")
        rhs_cands  = ctx.get("top_rhs_candidates", [])
        figma_sample = "\n".join((ctx.get("figma_text") or [])[:40])

        explain_prompt = f"""
Explain briefly and neutrally (3–5 sentences) how this header could be selected:
focus on shape (short, human-like phrase), overlap with RHS leaf names, and general similarity.

Header: {header}
Matched UI label: {matched_ui}
Top RHS candidates: {json.dumps(rhs_cands, ensure_ascii=False)}
Sample UI labels:
{figma_sample}
""".strip()

        try:
            exp = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": explain_prompt}])
            explanation = exp["message"]["content"]
        except Exception as oe:
            explanation = f"(Explanation unavailable: {oe})"

        save_feedback()

        return jsonify({
            "header": header,
            "status": status,
            "patterns_used": patterns,
            "explanation": explanation
        })

    except Exception as e:
        return jsonify({"error": "Feedback failed", "details": str(e)}), 500

# -------------------- Root --------------------
@app.get("/")
def home():
    return jsonify({
        "message": "POST /api/find_fields with {figma_json, data_json}. "
                   "Labels come from TEXT nodes first; if absent, we clean Figma 'name' fields. "
                   "If Figma is sparse, we enrich with short RHS aliases that include Figma tokens. "
                   "Matches are ranked lexically on RHS leaves (with schema aliases) and FAISS as backstop. "
                   "POST /api/feedback with {header, status}. Add ?debug=1 to inspect."
    })

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "5000")))
    args = parser.parse_args()
    print(f"✅ API running at http://localhost:{args.port}")
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        print("ℹ️  Ollama:", OLLAMA_URL, "ok" if r.status_code == 200 else f"status {r.status_code}")
    except Exception as e:
        print("ℹ️  Ollama:", OLLAMA_URL, f"error: {e}")
    app.run(host="0.0.0.0", port=args.port, debug=True)
