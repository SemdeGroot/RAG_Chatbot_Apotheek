#!/usr/bin/env python3
"""
Scraper voor apotheek.nl medicijnpagina’s (volwassenen + optioneel 'kindertekst').
- Werkt op URL of lokale HTML.
- Houdt content onder de juiste H2/H3 binnen het dichtstbijzijnde <li> (accordion).
- Filtert widgets/CTA’s.
- Optionele dedupe en lichte cleanup.
- Schrijft *_clean.json (optioneel aparte *_kindertekst_clean.json).

Voorbeeld:
  python -m apotheek_scraper.apotheek_scraper https://www.apotheek.nl/medicijnen/paracetamol --outdir data/clean_json --sleep 2 --include-children
"""
from __future__ import annotations
import argparse, re, sys, os
from typing import List, Dict, Any, Optional
from pathlib import Path
from bs4 import BeautifulSoup, Tag
from .utils import (
    fetch_url, read_local, write_json, is_url,
    basename_from_resource, RateLimiter, RobotsCache,
    DEFAULT_UA, derive_children_url
)

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

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()

def is_widget_title(title: str) -> bool:
    if not title:
        return True
    t = _norm(title)
    if t in {_norm(x) for x in SKIP_TITLES}:
        return True
    if any(x in t for x in ["webapotheker", "aanmelden", "inloggen", "nieuwsbrief"]):
        return True
    if len(t) <= 2:
        return True
    return False

def within(el: Tag, ancestor: Tag) -> bool:
    cur = el
    while cur and isinstance(cur, Tag):
        if cur is ancestor:
            return True
        cur = cur.parent
    return False

def nearest_container(h2: Tag) -> Tag:
    # liefst de <li> die het accordion-item vormt
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

def extract_section_from_h2(h2: Tag) -> Optional[Dict[str, Any]]:
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
                (current_sub or section)["blocks"].append({"type": "list", "ordered": el.name == "ol", "items": items})

    if not section["blocks"] and not any(s.get("blocks") for s in subsections):
        return None
    if subsections:
        section["subsections"] = [s for s in subsections if s.get("blocks")]
    return section

def _dedupe_blocks(section: Dict[str, Any]) -> None:
    """Verwijder paragraphs die exact dupliceren met list items binnen dezelfde (sub)section."""
    def dedupe_in(blocks: List[Dict[str, Any]]):
        list_items = []
        for b in blocks:
            if b.get("type") == "list":
                list_items.extend([_norm(x) for x in b.get("items", [])])
        if not list_items:
            return blocks
        deduped = []
        for b in blocks:
            if b.get("type") == "paragraph":
                if _norm(b.get("text", "")) in list_items:
                    continue
            deduped.append(b)
        return deduped

    section["blocks"] = dedupe_in(section.get("blocks", []))
    for sub in section.get("subsections", []) or []:
        sub["blocks"] = dedupe_in(sub.get("blocks", []))

def _merge_short_paragraphs(section: Dict[str, Any]) -> None:
    """Voeg opeenvolgende zeer korte paragrafen samen (cosmetisch)."""
    def merge(blocks: List[Dict[str, Any]]):
        out = []
        for b in blocks:
            if out and b.get("type") == "paragraph" and out[-1].get("type") == "paragraph":
                a = out[-1]["text"].strip()
                c = b["text"].strip()
                if len(a.split()) <= 4 or (not a.endswith((".", ":", "?", "!")) and len(c.split()) <= 30):
                    out[-1]["text"] = (a + " " + c).strip()
                    continue
            out.append(b)
        return out
    section["blocks"] = merge(section.get("blocks", []))
    for sub in section.get("subsections", []) or []:
        sub["blocks"] = merge(sub.get("blocks", []))

def parse_html(html: str, source_hint: str = "", *, dedupe: bool = True, merge_pars: bool = False) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "lxml")
    title = soup.find("h1").get_text(" ", strip=True) if soup.find("h1") else ""
    sections = []
    for h2 in soup.find_all("h2"):
        sec = extract_section_from_h2(h2)
        if sec:
            if dedupe:
                _dedupe_blocks(sec)
            if merge_pars:
                _merge_short_paragraphs(sec)
            sections.append(sec)
    return {"url": source_hint or None, "title": title, "sections": sections}

def save_clean_json(data: Dict[str, Any], outdir: Path, base: str, *, suffix: str = "") -> Path:
    name = f"{base}{suffix}_clean.json"
    out = outdir / name
    write_json(data, out)
    return out

def scrape_resource(resource: str, *, outdir: Path, sleep: float = 2.0,
                    include_children: bool = False, children_inline: bool = False,
                    dedupe: bool = True, merge_pars: bool = False,
                    user_agent: str = DEFAULT_UA, robots: Optional[RobotsCache] = None) -> List[Path]:
    """
    Scrape één resource (URL of lokale file).
    - include_children=True: probeer kindertekst-URL ook te halen.
    - children_inline=True: voeg kindertekst in dezelfde JSON (anders apart bestand).
    """
    out_paths: List[Path] = []
    outdir.mkdir(parents=True, exist_ok=True)
    rate = RateLimiter(min_interval_sec=sleep)
    robots = robots or RobotsCache()

    def get_html(res: str) -> str:
        if is_url(res):
            if robots.allowed(res, user_agent):
                return fetch_url(res, rate=rate, user_agent=user_agent)
            else:
                raise RuntimeError(f"Robots.txt blokkeert: {res}")
        return read_local(res)

    base = basename_from_resource(resource)
    html = get_html(resource)
    data = parse_html(html, source_hint=resource, dedupe=dedupe, merge_pars=merge_pars)

    if include_children and is_url(resource):
        kid_url = derive_children_url(resource)
        try:
            kid_html = get_html(kid_url)
            kid_data = parse_html(kid_html, source_hint=kid_url, dedupe=dedupe, merge_pars=merge_pars)
            if children_inline:
                # voeg kindersecties toe met prefix in titels
                for sec in kid_data.get("sections", []):
                    sec["title"] = f"[Kinderen] {sec.get('title','')}"
                    data["sections"].append(sec)
                out_path = save_clean_json(data, outdir, base)
                out_paths.append(out_path)
            else:
                # schrijf apart bestand
                out_paths.append(save_clean_json(data, outdir, base))
                out_paths.append(save_clean_json(kid_data, outdir, base, suffix="_kindertekst"))
                return out_paths
        except Exception:
            # Kindertekst niet gevonden/verboden → alleen adults opslaan
            pass

    out_paths.append(save_clean_json(data, outdir, base))
    return out_paths

def main():
    ap = argparse.ArgumentParser(description="Scrape apotheek.nl medicine pages naar *_clean.json")
    ap.add_argument("inputs", nargs="+", help="URLs of lokale HTML-bestanden")
    ap.add_argument("--outdir", default="data/clean_json", help="Uitvoer directory")
    ap.add_argument("--sleep", type=float, default=2.0, help="Delay (s) tussen requests (vriendelijk scrapen)")
    ap.add_argument("--include-children", action="store_true", help="Probeer ook de kindertekst-pagina op te halen")
    ap.add_argument("--children-inline", action="store_true", help="Voeg kindertekst in hetzelfde JSON-bestand toe")
    ap.add_argument("--no-dedupe", action="store_true", help="Zet dedupe (list vs paragraph) uit")
    ap.add_argument("--merge-paragraphs", action="store_true", help="Voeg korte opeenvolgende paragrafen samen")
    ap.add_argument("--user-agent", default=DEFAULT_UA, help="Custom User-Agent header")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    robots = RobotsCache()
    ok, err = 0, 0
    for res in args.inputs:
        try:
            paths = scrape_resource(
                res,
                outdir=outdir,
                sleep=args.sleep,
                include_children=args.include_children,
                children_inline=args.children_inline,
                dedupe=not args.no_dedupe,
                merge_pars=args.merge_paragraphs,
                user_agent=args.user_agent,
                robots=robots
            )
            for p in paths:
                print(f"[OK] {res} -> {p}")
            ok += 1
        except Exception as e:
            print(f"[ERR] {res}: {e}", file=sys.stderr)
            err += 1
    if err and not ok:
        sys.exit(1)

if __name__ == "__main__":
    main()