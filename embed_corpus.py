#!/usr/bin/env python3
"""Build framework_vectors.json — the embedding side of Ask SCL's hybrid retrieval.

Embeds every framework.json entry with nomic-embed-text via local Ollama,
normalizes to unit length, quantizes to int8, and writes a plain-JSON vector
file the Worker fetches alongside framework.json. Plain JSON int arrays on
purpose: no base64/atob path to test, Cloudflare gzips the wire size down,
and the eval harness can load the same file the same way.

RERUN THIS on every framework.json regeneration (version bumps): the Worker
refuses the vector file unless its version AND count AND ids match the live
framework.json, and silently falls back to keyword-only retrieval on mismatch.

Requires: Ollama serving on localhost:11434 with nomic-embed-text pulled.
Usage: python3 embed_corpus.py
"""
import json, urllib.request
from pathlib import Path

HERE = Path(__file__).parent
MODEL = "nomic-embed-text"
# nomic task prefixes: corpus side "search_document: ", query side
# "search_query: " (the Worker and eval runner must use the query prefix).
DOC_PREFIX = "search_document: "
BATCH = 16

fw = json.loads((HERE / "framework.json").read_text())
entries = fw["entries"]
# Keywords ride along in the embedded text on purpose: they are hand-authored
# visitor phrasings, and including them moved the worst paraphrase-bench gold
# rank from 13 to 1 (measured 2026-08-26; title+text alone was weaker).
texts = [DOC_PREFIX + e["title"] + "\n" + ", ".join(e.get("keywords", [])) + "\n" + e["text"]
         for e in entries]

vecs = []
for i in range(0, len(texts), BATCH):
    body = json.dumps({"model": MODEL, "input": texts[i:i + BATCH]}).encode()
    req = urllib.request.Request("http://localhost:11434/api/embed", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        vecs.extend(json.load(r)["embeddings"])
    print(f"embedded {min(i + BATCH, len(texts))}/{len(texts)}", flush=True)

dim = len(vecs[0])
assert all(len(v) == dim for v in vecs)

quant, scales = [], []
for v in vecs:
    norm = sum(x * x for x in v) ** 0.5 or 1.0
    unit = [x / norm for x in v]
    m = max(abs(x) for x in unit) or 1.0
    scale = m / 127.0
    quant.append([max(-127, min(127, round(x / scale))) for x in unit])
    scales.append(round(scale, 8))

out = {
    "model": MODEL,
    "query_prefix": "search_query: ",
    "version": fw.get("version", ""),
    "dim": dim,
    "count": len(entries),
    "ids": [e["id"] for e in entries],
    "scales": scales,
    "vecs": quant,
}
path = HERE / "framework_vectors.json"
path.write_text(json.dumps(out, separators=(",", ":")))
print(f"wrote {path.name}: {len(entries)} vectors x {dim} dims, "
      f"{path.stat().st_size // 1024}KB raw (gzip on the wire)")
