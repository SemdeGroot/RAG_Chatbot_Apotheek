#!/usr/bin/env python3
"""
Batch-scraper: leest een lijst met URLs (één per regel, '#' = comment) en
schrijft voor elke URL een *_clean.json (en optioneel _kindertekst_clean.json).
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
from typing import List
from .apotheek_scraper import scrape_resource
from .utils import RobotsCache

def load_urls(path: str | Path) -> List[str]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out

def main():
    ap = argparse.ArgumentParser(description="Batch scrape apotheek.nl URLs")
    ap.add_argument("--urls", required=True, help="Pad naar txt-bestand met een URL per regel")
    ap.add_argument("--out", default="data/clean_json", help="Uitvoer directory")
    ap.add_argument("--sleep", type=float, default=2.0, help="Delay (s) tussen requests")
    ap.add_argument("--include-children", action="store_true", help="Kindertekst ook ophalen")
    ap.add_argument("--children-inline", action="store_true", help="Kindertekst inline toevoegen")
    args = ap.parse_args()

    urls = load_urls(args.urls)
    outdir = Path(args.out)
    robots = RobotsCache()
    for u in urls:
        try:
            paths = scrape_resource(
                u, outdir=outdir, sleep=args.sleep,
                include_children=args.include_children,
                children_inline=args.children_inline,
                robots=robots
            )
            for p in paths:
                print(f"[OK] {u} -> {p}")
        except Exception as e:
            print(f"[ERR] {u}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
