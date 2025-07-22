from flask import Flask, jsonify
from pathlib import Path
import json, re

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.embeddings import OllamaEmbeddings

app = Flask(__name__)

# ───────────── Load JSON Files ─────────────
lhs_path = Path("data/FigmaOpportunityTable.json")   # ← left-hand UI JSON
rhs_path = Path("data/DataRightHS.json")             # ← your uploaded right-hand JSON

lhs_data = json.loads(lhs_path.read_text(encoding="utf-8"))
rhs_data = json.loads(rhs_path.read_text(encoding="utf-8"))

# ───────────── Step 1: Extract Top 10 UI Headers ─────────────
def extract_likely_headers(figma_json: dict) -> list[str]:
    out = []

    def is_numeric(t: str) -> bool:
        cleaned = t.replace(",", "").replace("%", "").replace("$", "").strip()
        return cleaned.replace(".", "").isdigit()

    def is_likely_header(txt: str) -> bool:
    return (
        txt
        and not is_numeric(txt)
        and len(txt) <= 30
        and not any(c in txt for c in "@%/:()[]0123456789$")
    )

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "TEXT":
                txt = node.get("characters", "").strip()
                if is_likely_header(txt):
                    out.append(txt)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(figma_json)
    deduped = list(dict.fromkeys(out))
    return deduped[:10]

# ───────────── Step 2: Match Headers to RHS Fields ─────────────
def match_headers_to_schema(headers: list[str], rhs_schema: dict) -> dict:
    documents = []
    for key, value in rhs_schema.items():
        if not value:  # skip empty or null fields
            continue
        combined = f"{key}: {value}"
        documents.append(Document(page_content=combined, metadata={"field": key}))

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_documents(documents, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", k=3)

    results = {}
    for header in headers:
        docs = retriever.invoke(header)
        results[header] = [doc.metadata["field"] for doc in docs]

    return results

# ───────────── API Endpoint ─────────────
@app.route("/api/match-fields", methods=["GET"])
def match_fields():
    headers = extract_likely_headers(lhs_data)
    matches = match_headers_to_schema(headers, rhs_data)
    return jsonify(matches)

# ───────────── Run Flask Server ─────────────
if __name__ == "__main__":
    app.run(debug=True)