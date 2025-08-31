#!/usr/bin/env python3
"""
RAG-chat: haalt context uit je FAISS DB en vraagt de Groq-chatbot om een NL-antwoord met bronverwijzing.

Gebruik:
  python -m augmented_generation.rag_chat --db data/vectordb --k 5 --q "Wanneer mag ik ibuprofen gebruiken tijdens zwangerschap?"
  # Interactieve chat (met dezelfde DB):
  python -m augmented_generation.rag_chat --db data/vectordb --interactive
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
from typing import List, Dict, Tuple

from embedding.common import read_index_and_meta, embed_query
from .providers.groq_client import chat as groq_chat

DEFAULT_MODEL = "llama-3.3-70b-versatile"

def retrieve(db_dir: Path, query: str, k: int = 5) -> List[Tuple[float, Dict]]:
    index, metas, cfg = read_index_and_meta(db_dir)
    q_vec = embed_query(query, cfg.get("model_name"))
    scores, idxs = index.search(q_vec, k)
    hits = []
    for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
        if idx == -1: 
            continue
        hits.append((float(score), metas[idx]))
    return hits

def build_context_blocks(hits: List[Tuple[float, Dict]]) -> List[str]:
    blocks = []
    for i, (score, m) in enumerate(hits, 1):
        place = f"{m.get('title','').strip()} > {m.get('section','').strip()}"
        if m.get("subsection"):
            place += f" > {m['subsection'].strip()}"
        url = m.get("url") or m.get("source_file") or ""
        txt = (m.get("raw_text","") or "").strip()
        block = f"[{i}] {place}\n{txt}\nURL: {url}"
        blocks.append(block)
    return blocks

def make_messages(question: str, context_blocks: List[str]) -> List[Dict[str, str]]:
    """
    Prompt die GEEN inline citaties en GEEN 'Bronnen:'-regel laat genereren.
    De applicatie toont zelf de bronnenlijst onder het antwoord.
    """
    context = "\n\n".join(context_blocks)
    system = (
        "Je bent een assistent die antwoorden geeft over geneesmiddelen op basis van meegeleverde context. "
        "Gebruik uitsluitend die context; verzin niets. Antwoord beknopt, helder en feitelijk in de taal van de vraag; als die onduidelijk is, antwoord in het Nederlands. "
        "Noem doseringen/contra-indicaties alleen als die expliciet in de context staan. "
        "BELANGRIJK: plaats GEEN inline verwijzingen ([1], [2], e.d.) en voeg GEEN bronnen- of referentieblok toe. "
        "Schrijf uitsluitend het antwoord in lopende tekst."
    )
    user = (
        f"VRAAG:\n{question}\n\n"
        "CONTEXT (passages, eventueel genummerd door het systeem):\n"
        f"{context}\n\n"
        "INSTRUCTIES:\n"
        "- Beantwoord uitsluitend op basis van de context.\n"
        "- Schrijf alleen het antwoord; voeg GEEN citaties of 'Bronnen:'-regel toe.\n"
        "- Als informatie ontbreekt, zeg dat expliciet."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

def answer_question(db_dir: Path, question: str, k: int = 5,
                    model: str = DEFAULT_MODEL, max_tokens: int = 600,
                    temperature: float = 0.2, stream: bool = False) -> str:
    hits = retrieve(db_dir, question, k=k)
    if not hits:
        return "Geen context gevonden in de vector database. Bouw eerst de index of pas je vraag aan."
    blocks = build_context_blocks(hits)
    messages = make_messages(question, blocks)
    reply = groq_chat(messages, model=model, max_tokens=max_tokens, temperature=temperature, stream=stream)
    return reply

def _print_sources(hits: List[Tuple[float, Dict]], answer: str):
    used = []
    for i in range(1, len(hits)+1):
        if f"[{i}]" in answer:
            used.append(i)
    if not used:
        used = list(range(1, min(4, len(hits))+1))
    print("\nBRONNEN:")
    for i in used:
        _, m = hits[i-1]
        place = f"{m.get('title','')} > {m.get('section','')}"
        if m.get("subsection"):
            place += f" > {m['subsection']}"
        url = m.get("url") or m.get("source_file") or ""
        print(f"[{i}] {place} — {url}")

def main():
    ap = argparse.ArgumentParser(description="GROQ RAG chatbot (NL) op FAISS-vectorDB")
    ap.add_argument("--db", required=True, help="Map met index.faiss/meta.jsonl/config.json")
    ap.add_argument("--q", help="Eenmalige vraag (NL)")
    ap.add_argument("--k", type=int, default=5, help="Top-K passages")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--stream", action="store_true", help="Stream token-voor-token naar stdout")
    ap.add_argument("--interactive", action="store_true", help="Interatieve chat-lus")
    args = ap.parse_args()

    db_dir = Path(args.db)

    if args.interactive:
        print("Interactieve RAG-chat. Typ je vraag; Ctrl+C om te stoppen.\n")
        while True:
            try:
                q = input("❓ Vraag: ").strip()
                if not q:
                    continue
                hits = retrieve(db_dir, q, k=args.k)
                if not hits:
                    print("Geen context gevonden.")
                    continue
                blocks = build_context_blocks(hits)
                reply = groq_chat(make_messages(q, blocks),
                                  model=args.model,
                                  max_tokens=args.max_tokens,
                                  temperature=args.temperature,
                                  stream=args.stream)
                print("\n💬 Antwoord:\n" + reply.strip())
                _print_sources(hits, reply)
                print()
            except KeyboardInterrupt:
                print("\nBye!")
                sys.exit(0)
    else:
        if not args.q:
            print("Geef een vraag met --q of gebruik --interactive.", file=sys.stderr)
            sys.exit(1)
        hits = retrieve(db_dir, args.q, k=args.k)
        if not hits:
            print("Geen context gevonden in de vector database.", file=sys.stderr)
            sys.exit(1)
        blocks = build_context_blocks(hits)
        reply = groq_chat(make_messages(args.q, blocks),
                          model=args.model,
                          max_tokens=args.max_tokens,
                          temperature=args.temperature,
                          stream=args.stream)
        print("\n=== ANTWOORD ===\n")
        print(reply.strip())
        print()
        _print_sources(hits, reply)

if __name__ == "__main__":
    main()
