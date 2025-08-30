#!/usr/bin/env python3
"""
Kleine Groq client-wrapper (OpenAI-achtige SDK).
- Leest GROQ_API_KEY uit env of .env (indien aanwezig).
- Biedt chat() met (optioneel) streaming.
"""
from __future__ import annotations
import os, re
from pathlib import Path
from typing import List, Dict, Optional

def _load_env_if_exists(path: str = ".env") -> None:
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
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]
        os.environ.setdefault(key, val)

def get_groq(api_key: Optional[str] = None):
    _load_env_if_exists(".env")
    from groq import Groq  # lazy import
    key = api_key or os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY ontbreekt. Zet 'setx GROQ_API_KEY \"sk_...\"' (Windows) of export in je shell/.env.")
    return Groq(api_key=key)

def chat(messages: List[Dict[str, str]],
         model: str = "llama-3.3-70b-versatile",
         max_tokens: int = 600,
         temperature: float = 0.2,
         stream: bool = False,
         api_key: Optional[str] = None) -> str:
    client = get_groq(api_key=api_key)
    if not stream:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content
    # Streaming
    out = []
    with client.chat.completions.stream(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    ) as stream_resp:
        for event in stream_resp:
            if event.event == "completion.delta" and event.delta and event.delta.content:
                chunk = event.delta.content
                out.append(chunk)
                print(chunk, end="", flush=True)
        # newline for neatness
        print()
        return "".join(out)
