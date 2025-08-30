#!/usr/bin/env python3
"""
Run example scrapes for apotheek.nl and write to data/clean_json.

Standaardgedrag (zonder args):
- Scrapt voorbeeld-URLs (paracetamol, ibuprofen, metoprolol)
- Schrijft *_clean.json naar data/clean_json/
- Vriendelijk rate-limiten (sleep=2s)
- Als apotheek_scraper/urls_example.txt bestaat, gebruik die lijst i.p.v. defaults

Gebruik (aanbevolen):
  python -m apotheek_scraper.run_examples

Met opties:
  python -m apotheek_scraper.run_examples --outdir data/clean_json --sleep 2 --include-children
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
from typing import List

from .apotheek_scraper import scrape_resource
from .utils import RobotsCache

# Default voorbeeld-URLs (gebruikt als er geen urls_example.txt is)
DEFAULT_URLS: List[str] = [
    "https://www.apotheek.nl/medicijnen/paracetamol",
    "https://www.apotheek.nl/medicijnen/ibuprofen",
    "https://www.apotheek.nl/medicijnen/metoprolol",
]

def load_urls_from_file(fpath: Path) -> List[str]:
    """Lees een URL-lijst (één per regel, '#' = comment)."""
    if not fpath.exists():
        return DEFAULT_URLS
    urls = []
    for line in fpath.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line)
    return urls or DEFAULT_URLS

def main():
    ap = argparse.ArgumentParser(description="Scrape example apotheek.nl medicine pages → *_clean.json")
    ap.add_argument("--outdir", default="data/clean_json", help="Uitvoer directory (default: data/clean_json)")
    ap.add_argument("--sleep", type=float, default=2.0, help="Delay (s) tussen requests (vriendelijk scrapen)")
    ap.add_argument("--include-children", action="store_true", help="Probeer ook de kindertekst-pagina op te halen")
    ap.add_argument("--children-inline", action="store_true", help="Voeg kindertekst in hetzelfde JSON-bestand toe")
    args = ap.parse_args()

    urls_file = Path(__file__).with_name("urls_example.txt")
    urls = load_urls_from_file(urls_file)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    robots = RobotsCache()
    ok, err = 0, 0
    for u in urls:
        try:
            paths = scrape_resource(
                u,
                outdir=outdir,
                sleep=args.sleep,
                include_children=args.include_children,
                children_inline=args.children_inline,
                robots=robots
            )
            for p in paths:
                print(f"[OK] {u} -> {p}")
            ok += 1
        except Exception as e:
            print(f"[ERR] {u}: {e}", file=sys.stderr)
            err += 1

    if err and not ok:
        sys.exit(1)

if __name__ == "__main__":
    main()

# RUN below command to scrape example urls 
# python -m apotheek_scraper.run_examples