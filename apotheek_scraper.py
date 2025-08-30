#!/usr/bin/env python3
"""
Scraper voor apotheek.nl medicijnpagina’s.
- Werkt met URLs (en ook lokale HTML-bestanden).
- Schrijft per input een *_clean.json met nette secties en subsections.
"""

import argparse, json, re, sys, time
from pathlib import Path
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup, Tag, NavigableString

SKIP_TITLES = {
    "Vind een apotheek",
    "Vraag het de webapotheker",
    "Disclaimer",
    "Nieuws",
    "Meer over",
    "Meer informatie",
    "Gerelateerde onderwerpen",
    "Over deze site",
    "Veelgestelde vragen",
}

UA = {"User-Agent": "apotheek-scraper/1.0 (+contact: semdegroot2003@gmail.com)"}


def is_widget_title(title: str) -> bool:
    if not title:
        return True
    t = title.strip().lower()
    if t in {s.lower() for s in SKIP_TITLES}:
        return True
    if any(x in t for x in ["webapotheker", "aanmelden", "inloggen", "nieuwsbrief"]):
        return True
    if len(t) <= 2:
        return True
    return False


def read_input(resource: str, sleep_seconds: float = 0.0) -> str:
    """Lees lokale file of fetch een URL."""
    if re.match(r"^https?://", resource, re.I):
        if sleep_seconds:
            time.sleep(sleep_seconds)
        resp = requests.get(resource, headers=UA, timeout=30)
        resp.raise_for_status()
        return resp.text
    return Path(resource).read_text(encoding="utf-8", errors="ignore")


def within(el: Tag, ancestor: Tag) -> bool:
    cur = el
    while cur and isinstance(cur, Tag):
        if cur is ancestor:
            return True
        cur = cur.parent
    return False


def nearest_container(h2: Tag) -> Tag:
    cur = h2
    while cur and isinstance(cur, Tag):
        if cur.name == "li":
            return cur
        cur = cur.parent
    for tag in ("section", "article", "main", "body"):
        anc = h2.find_parent(tag)
        if anc:
            return anc
    return h2.parent or h2


def iter_until_container_end(start: Tag, container: Tag):
    el = start.next_element
    while el:
        if isinstance(el, Tag):
            if not within(el, container):
                break
            yield el
        el = el.next_element


def extract_section_from_h2(h2: Tag) -> dict | None:
    title = h2.get_text(" ", strip=True)
    if is_widget_title(title):
        return None

    container = nearest_container(h2)
    section = {"title": title, "blocks": []}
    subsections = []
    current_sub = None

    for el in iter_until_container_end(h2, container):
        if el.name == "h2":
            break
        if el.name == "h3":
            sub_title = el.get_text(" ", strip=True)
            if is_widget_title(sub_title):
                current_sub = None
                continue
            current_sub = {"title": sub_title, "blocks": []}
            subsections.append(current_sub)
            continue

        if el.name == "p":
            txt = el.get_text(" ", strip=True)
            if txt:
                (current_sub or section)["blocks"].append({"type": "paragraph", "text": txt})
        elif el.name in ("ul", "ol"):
            items = []
            for li in el.find_all("li", recursive=False):
                if li.find(["h1", "h2", "h3", "h4"]):
                    continue
                itxt = li.get_text(" ", strip=True)
                if itxt:
                    items.append(itxt)
            if items:
                (current_sub or section)["blocks"].append(
                    {"type": "list", "ordered": el.name == "ol", "items": items}
                )

    if not section["blocks"] and not any(s.get("blocks") for s in subsections):
        return None
    if subsections:
        section["subsections"] = [s for s in subsections if s.get("blocks")]
    return section


def parse_html(html: str, source_hint: str = "") -> dict:
    soup = BeautifulSoup(html, "lxml")
    title = soup.find("h1").get_text(" ", strip=True) if soup.find("h1") else ""
    sections = []
    for h2 in soup.find_all("h2"):
        sec = extract_section_from_h2(h2)
        if sec:
            sections.append(sec)
    return {"url": source_hint or None, "title": title, "sections": sections}


def derive_basename(resource: str) -> str:
    if re.match(r"^https?://", resource, re.I):
        path = urlparse(resource).path.rstrip("/")
        name = Path(path).name or "index"
        return name
    return Path(resource).stem


def main():
    ap = argparse.ArgumentParser(description="Scrape apotheek.nl medicine pages to structured JSON.")
    ap.add_argument("inputs", nargs="+", help="URLs of apotheek.nl medicine pages (or local HTML files)")
    ap.add_argument("--outdir", default=".", help="Output directory for *_clean.json files")
    ap.add_argument("--sleep", type=float, default=0.0, help="Delay (seconds) between URL fetches")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for res in args.inputs:
        try:
            html = read_input(res, sleep_seconds=args.sleep)
            data = parse_html(html, source_hint=res)
            base = derive_basename(res)
            outpath = outdir / f"{base}_clean.json"
            outpath.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[OK] {res} -> {outpath}")
        except Exception as e:
            print(f"[ERR] {res}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
