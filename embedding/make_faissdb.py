#!/usr/bin/env python3
"""
Compatibele wrapper zodat bestaande commando's blijven werken:

Build:
  python embedding/make_faissdb.py build --input-dir data/clean_json --pattern "*_clean.json" --outdir data/vectordb --batch-size 32

Query:
  python embedding/make_faissdb.py query --db data/vectordb --k 5 --q "Wat is de maximale dosering paracetamol per dag?"
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
from .build_index import main as build_main
from .query_index import main as query_main

def main():
    ap = argparse.ArgumentParser(description="FAISS DB builder/query (compat)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="Bouw index (proxy naar embedding.build_index)")
    p_build.add_argument("--input-dir", required=True)
    p_build.add_argument("--pattern", default="*_clean.json")
    p_build.add_argument("--outdir", required=True)
    p_build.add_argument("--batch-size", type=int, default=64)
    p_build.add_argument("--model", default=None)
    p_build.add_argument("--dedupe", action="store_true")

    p_query = sub.add_parser("query", help="Query index (proxy naar embedding.query_index)")
    p_query.add_argument("--db", required=True)
    p_query.add_argument("--k", type=int, default=5)
    p_query.add_argument("--q", required=True)
    p_query.add_argument("--show", type=int, default=500)
    p_query.add_argument("--scores", action="store_true")

    args, extra = ap.parse_known_args()

    if args.cmd == "build":
        # Routeer args naar build_index
        sys.argv = ["-m", "embedding.build_index",
                    "--input-dir", args.input_dir,
                    "--pattern", args.pattern,
                    "--outdir", args.outdir,
                    "--batch-size", str(args.batch_size)]
        if args.model:
            sys.argv += ["--model", args.model]
        if args.dedupe:
            sys.argv += ["--dedupe"]
        return build_main()

    if args.cmd == "query":
        sys.argv = ["-m", "embedding.query_index",
                    "--db", args.db,
                    "--k", str(args.k),
                    "--q", args.q,
                    "--show", str(args.show)]
        if args.scores:
            sys.argv += ["--scores"]
        return query_main()

if __name__ == "__main__":
    main()

