#!/usr/bin/env python3
"""
Bouw een FAISS vector DB uit *_clean.json bestanden.
Voorbeeld:
  python -m embedding.build_index --input-dir data/clean_json --pattern "*_clean.json" --outdir data/vectordb --batch-size 32
"""
from __future__ import annotations
import argparse, sys, os, json
from pathlib import Path
from typing import List, Dict, Any
from .common import (
    DEFAULT_MODEL,
    load_clean_json_files,
    load_json,
    flatten_chunks_from_clean_json,
    build_embeddings,
    write_index,
)

def main():
    ap = argparse.ArgumentParser(description="Build FAISS index from *_clean.json files")
    ap.add_argument("--input-dir", required=True, help="Map met *_clean.json")
    ap.add_argument("--pattern", default="*_clean.json", help="Glob pattern (default: *_clean.json)")
    ap.add_argument("--outdir", required=True, help="Doelmap voor index.faiss/meta.jsonl/config.json")
    ap.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    ap.add_argument("--model", default=os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL), help="Embedding modelnaam")
    ap.add_argument("--dedupe", action="store_true", help="Verwijder exacte duplicaat-chunks (same text)")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    files = load_clean_json_files(input_dir, args.pattern)
    if not files:
        print(f"Geen bestanden in {input_dir} voor pattern {args.pattern}", file=sys.stderr)
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

    if args.dedupe:
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
        print("Geen chunks geproduceerd; abort.", file=sys.stderr)
        sys.exit(1)

    texts = [c["text"] for c in all_chunks]
    embs = build_embeddings(texts, model_name=args.model, batch_size=args.batch_size)
    write_index(Path(args.outdir), embs, all_chunks, args.model)
    print(f"[OK] index gebouwd uit {len(files)} bestanden, {len(all_chunks)} chunks.")

if __name__ == "__main__":
    main()
