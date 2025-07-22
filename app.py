from flask import Flask, request, jsonify
from pathlib import Path
import json, re

app = Flask(__name__)

# ───────────── Load Figma UI JSON ─────────────
lhs_path = Path("data/FigmaOpportunityTable.json")
lhs_data = json.loads(lhs_path.read_text(encoding="utf-8"))

# ───────────── Extract Likely Headers ─────────────
def extract_likely_headers(figma_json: dict) -> list[str]:
    out = []

    def is_numeric(t: str) -> bool:
        cleaned = t.replace(",", "").replace("%", "").replace("$", "").strip()
        return cleaned.replace(".", "").isdigit()

    def is_likely_header(txt: str) -> bool:
        return (
            txt
            and not is_numeric(txt)
            and 1 <= len(txt.split()) <= 3
            and txt[0].isupper()
            and not any(c in txt for c in "-@%/:()[]0123456789$•")
            and re.match(r"^[A-Z][a-zA-Z ]+$", txt)
        )

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "TEXT":
                txt = node.get("characters", "").strip()
                if is_likely_header(txt):
                    out.append(txt)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(figma_json)

    deduped = list(dict.fromkeys(out))  # Remove duplicates, keep order
    return deduped[:10]  # Return top 10

# ───────────── API Endpoint ─────────────
@app.route("/api/top10", methods=["GET"])
def get_top10_headers():
    headers = extract_likely_headers(lhs_data)
    return jsonify({"top_10_headers": headers})

# ───────────── Run Flask App ─────────────
if __name__ == "__main__":
    app.run(debug=True)