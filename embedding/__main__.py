#!/usr/bin/env python3
"""
One-command builder for your vector DB.

Gebruik:
  # Zonder opties (pakt defaults / .env):
  python -m embedding

  # Optioneel met opties:
  python -m embedding build --input-dir data/clean_json --outdir data/vectordb --pattern "*_clean.json" --batch-size 32 --dedupe
  python -m embedding query --db data/vectordb --k 5 --q "Wanneer mag ik ibuprofen gebruiken tijdens zwangerschap?"
"""
from __future__ import annotations
import argparse, os, sys, re
from pathlib import Path
from typing import List, Dict, Any

from .common import (
    DEFAULT_MODEL,
    load_clean_json_files,
    load_json,
    flatten_chunks_from_clean_json,
    build_embeddings,
    write_index,
    read_index_and_meta,
    embed_query,
)

# --- Mini .env loader (geen extra dependency nodig) ---
def load_env_if_exists(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)=(.*)$', line)
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip()
        # strip quotes
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]
        os.environ.setdefault(key, val)

def cmd_build(args):
    input_dir = Path(args.input_dir or os.getenv("CLEAN_JSON_DIR", "data/clean_json"))
    outdir    = Path(args.outdir    or os.getenv("VECTORDB_DIR",   "data/vectordb"))
    pattern   = args.pattern or os.getenv("CLEAN_PATTERN", "*_clean.json")
    model     = args.model or os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL)
    batch     = int(args.batch_size or os.getenv("EMBED_BATCH", "64"))
    dedupe    = args.dedupe or os.getenv("EMBED_DEDUPE", "1").lower() not in ("0","false","no")

    files = load_clean_json_files(input_dir, pattern)
    if not files:
        print(f"[ERR] Geen bestanden in {input_dir} met pattern '{pattern}'.", file=sys.stderr)
        sys.exit(1)

    all_chunks: List[Dict[str, Any]] = []
    for fp in files:
        try:
            doc = load_json(fp)
        except Exception as e:
            print(f"[WARN] overslaan {fp.name}: {e}", file=sys.stderr)
            continue
        chunks = flatten_chunks_from_clean_json(doc, default_url=str(doc.get('url') or fp.name))
        for ch in chunks:
            ch["source_file"] = fp.name
        all_chunks.extend(chunks)

    if dedupe:
        seen = set()
        deduped = []
        for ch in all_chunks:
            key = ch["text"]
            if key in seen:
                continue
            seen.add(key)
            deduped.append(ch)
        all_chunks = deduped

    if not all_chunks:
        print("[ERR] Geen chunks geproduceerd; abort.", file=sys.stderr)
        sys.exit(1)

    texts = [c["text"] for c in all_chunks]
    embs  = build_embeddings(texts, model_name=model, batch_size=batch)
    write_index(outdir, embs, all_chunks, model)
    print(f"[OK] index gebouwd → {outdir}  (files: {len(files)}, chunks: {len(all_chunks)})")

def cmd_query(args):
    db = Path(args.db or os.getenv("VECTORDB_DIR", "data/vectordb"))
    k  = int(args.k or 5)
    q  = args.q
    if not q:
        print("[ERR] Geef een query met --q", file=sys.stderr)
        sys.exit(1)

    index, metas, cfg = read_index_and_meta(db)
    model_name = cfg.get("model_name", DEFAULT_MODEL)
    q_vec = embed_query(q, model_name)
    scores, idxs = index.search(q_vec, k)

    print()
    for i, (score, idx) in enumerate(zip(scores[0].tolist(), idxs[0].tolist()), 1):
        if idx == -1:
            continue
        m = metas[idx]
        loc = f"{m.get('title','')} > {m.get('section','')}"
        if m.get("subsection"):
            loc += f" > {m['subsection']}"
        print(f"[{i}] score={score:.4f}  {loc}")
        print((m.get("raw_text","")[:500]).strip())
        if m.get("url"): print("url:", m["url"])
        print()

def build_parser():
    ap = argparse.ArgumentParser(description="One-command FAISS builder / simple query")
    sub = ap.add_subparsers(dest="cmd")

    p_build = sub.add_parser("build", help="Bouw de index (default)")
    p_build.add_argument("--input-dir")
    p_build.add_argument("--outdir")
    p_build.add_argument("--pattern")
    p_build.add_argument("--batch-size", type=int)
    p_build.add_argument("--model")
    p_build.add_argument("--dedupe", action="store_true")
    p_build.set_defaults(func=cmd_build)

    p_query = sub.add_parser("query", help="Query de index")
    p_query.add_argument("--db")
    p_query.add_argument("--q")
    p_query.add_argument("--k", type=int, default=5)
    p_query.set_defaults(func=cmd_query)

    return ap

def main():
    load_env_if_exists(".env")
    parser = build_parser()
    # Als je geen args geeft → doe "build" met defaults
    if len(sys.argv) == 1:
        args = parser.parse_args(["build"])
    else:
        args = parser.parse_args()
    if not hasattr(args, "func"):
        # fallback naar build
        args = parser.parse_args(["build"])
    args.func(args)

if __name__ == "__main__":
    main()

#Use below command to build FAISS database embeddings
#python -m embedding