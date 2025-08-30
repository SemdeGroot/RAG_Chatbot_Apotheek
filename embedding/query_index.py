#!/usr/bin/env python3
"""
Query een bestaande FAISS index.
Voorbeeld:
  python -m embedding.query_index --db data/vectordb --k 5 --q "Wanneer mag ik ibuprofen gebruiken tijdens zwangerschap?"
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
from typing import List, Tuple, Dict, Any
from .common import read_index_and_meta, embed_query

def main():
    ap = argparse.ArgumentParser(description="Query FAISS index")
    ap.add_argument("--db", required=True, help="Map met index.faiss/meta.jsonl/config.json")
    ap.add_argument("--q", required=True, help="Zoekvraag")
    ap.add_argument("--k", type=int, default=5, help="Top-K")
    ap.add_argument("--show", type=int, default=500, help="Max chars tonen per passage")
    ap.add_argument("--scores", action="store_true", help="Scores tonen")
    args = ap.parse_args()

    db = Path(args.db)
    index, metas, cfg = read_index_and_meta(db)
    model_name = cfg.get("model_name")
    q_vec = embed_query(args.q, model_name)
    scores, idxs = index.search(q_vec, args.k)

    print()  # lege regel
    for i, (score, idx) in enumerate(zip(scores[0].tolist(), idxs[0].tolist()), 1):
        if idx == -1: 
            continue
        m = metas[idx]
        loc = f"{m.get('title','')} > {m.get('section','')}"
        if m.get("subsection"):
            loc += f" > {m['subsection']}"
        s = f"[{i}] "
        if args.scores:
            s += f"score={score:.4f}  "
        s += loc
        print(s)
        print((m.get("raw_text","")[:args.show]).strip())
        if m.get("url"): print("url:", m["url"])
        print()

if __name__ == "__main__":
    main()
