#!/usr/bin/env python3
"""
Flask RAG chatbot (GROQ) met donkere UI.
- Leest je FAISS DB uit VECTORDB_DIR (default: data/vectordb)
- Leest GROQ_API_KEY uit .env of omgeving (augmented_generation/providers/groq_client.py doet dit ook)
- Chatgeschiedenis in Flask session (niet persistent)

Run:
  pip install flask groq sentence-transformers faiss-cpu
  python app.py
"""

from __future__ import annotations
import os, re, secrets
from pathlib import Path
from typing import List, Dict, Tuple
from flask import Flask, request, render_template_string, session, redirect, url_for
from flask_session import Session

# Kleine .env loader (zonder extra dependency)
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
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]
        os.environ.setdefault(key, val)

load_env_if_exists(".env")

# Imports uit je bestaande modules
from augmented_generation.rag_chat import retrieve, build_context_blocks, make_messages
from augmented_generation.providers.groq_client import chat as groq_chat

# Config
VECTORDB_DIR = Path(os.getenv("VECTORDB_DIR", "data/vectordb"))
RAG_MODEL    = os.getenv("RAG_MODEL", "llama-3.3-70b-versatile")
TOP_K        = int(os.getenv("RAG_TOPK", "5"))

app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me"),
    SESSION_TYPE="filesystem",
    SESSION_PERMANENT=False,
    SESSION_FILE_DIR=str(Path("./.flask_session").resolve()),
)
Path("./.flask_session").mkdir(parents=True, exist_ok=True)
Session(app)

HTML = r"""
<!doctype html>
<html lang="nl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RAG Apotheek Chatbot</title>
<style>
  :root{
    --bg0:#05070d;
    --bg1:#0b1220;
    --bg2:#0a0f1a;
    --ink:#dbe4ff;
    --muted:#9fb0d0;
    --brand:#2b5cff;
    --brand-2:#0e1a3a;
    --accent:#94b0ff;
    --danger:#ff5c5c;
    --ok:#2ee6a8;
  }
  *{box-sizing:border-box}
  html,body{height:100%;}
  body{
    margin:0;
    font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol";
    color:var(--ink);
    background:
      radial-gradient(1200px 600px at 80% -10%, #0f1a33 0%, transparent 60%),
      radial-gradient(800px 400px at 10% 110%, #0a1633 0%, transparent 60%),
      linear-gradient(180deg, var(--bg0) 0%, var(--bg1) 60%, #000 100%);
  }
  .wrap{
    max-width: 1000px;
    margin: 0 auto;
    padding: 24px 16px 80px;
  }
  .header{
    position: sticky; top:0; z-index:10;
    padding: 12px 16px;
    margin: -24px -16px 16px;
    background: linear-gradient(180deg, rgba(0,0,0,0.65), rgba(0,0,0,0.25));
    backdrop-filter: blur(6px);
    border-bottom: 1px solid rgba(255,255,255,0.06);
  }
  .title{
    display:flex; align-items:center; gap:12px;
    font-weight:700; letter-spacing:0.3px;
  }
  .dot{
    width:12px; height:12px; border-radius:50%;
    background: linear-gradient(135deg, var(--brand), var(--ok));
    box-shadow: 0 0 18px rgba(43,92,255,0.6);
  }
  .meta{
    color:var(--muted); font-size:13px;
  }
  .card{
    background: linear-gradient(180deg, rgba(14,26,58,0.55), rgba(5,7,13,0.5));
    border: 1px solid rgba(148,176,255,0.12);
    border-radius: 16px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.04);
  }
  form{
    display:flex; gap:10px; align-items: stretch;
    padding: 14px; border-bottom: 1px solid rgba(255,255,255,0.06);
  }
  input[type="text"]{
    flex:1;
    background: #0a1020;
    border: 1px solid rgba(148,176,255,0.2);
    color: var(--ink);
    padding: 12px 14px;
    border-radius: 10px;
    outline:none;
    transition: border .2s ease, box-shadow .2s ease;
  }
  input[type="text"]::placeholder{ color:#7f8eb0; }
  input[type="text"]:focus{
    border-color: var(--brand);
    box-shadow: 0 0 0 3px rgba(43,92,255,0.25);
  }
  button{
    background: linear-gradient(180deg, var(--brand), #2749c7);
    color: white; border: none; padding: 12px 18px;
    border-radius: 10px; font-weight:600; cursor:pointer;
    box-shadow: 0 8px 20px rgba(43,92,255,0.35);
  }
  .danger{ background: linear-gradient(180deg, #d94848, #b33636); box-shadow: 0 8px 20px rgba(217,72,72,0.35);}
  .row{ display:flex; gap: 12px; flex-wrap: wrap; align-items:center; }
  .grow{ flex:1; }
  .chat{
    padding: 6px 14px 14px;
    max-height: calc(100vh - 260px);
    overflow: auto;
  }
  .msg{
    display:flex; gap:10px; margin:14px 0;
  }
  .bubble{
    padding:12px 14px; border-radius: 12px; line-height:1.5;
    border:1px solid rgba(255,255,255,0.06);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
  }
  .user .bubble{
    background: #0f1b35; color: #e7ecff; border-color: rgba(148,176,255,0.18);
  }
  .bot .bubble{
    background: #0b0f1a; color: #dbe4ff; border-color: rgba(148,176,255,0.14);
  }
  .sources{
    margin-top:8px; font-size: 13px; color: var(--muted);
    border-top:1px dashed rgba(148,176,255,0.18); padding-top:8px;
  }
  .badge{
    display:inline-block; padding:2px 8px; border-radius:20px;
    background: #0f1b35; color: #a8b8e8; border:1px solid rgba(148,176,255,0.2);
    font-size:12px; margin-right:6px;
  }
  .footer{
    margin-top:14px; font-size:12px; color:#91a1c5; text-align:center;
  }
  a{ color: var(--accent); text-decoration: none; }
  a:hover{ text-decoration: underline; }
</style>
<script>
  window.addEventListener('load', () => {
    const box = document.querySelector('.chat');
    if (box) box.scrollTop = box.scrollHeight;
    const input = document.querySelector('#question');
    if (input) input.focus();
  });
</script>
</head>
<body>
  <div class="wrap">
    <div class="header">
      <div class="row">
        <div class="title"><div class="dot"></div> RAG Apotheek Chatbot</div>
        <div class="grow"></div>
        <div class="meta">
          DB: <span class="badge">{{ db_dir }}</span>
          Top-K: <span class="badge">{{ top_k }}</span>
          Model: <span class="badge">{{ model }}</span>
        </div>
      </div>
    </div>

    <div class="card">
      <form method="post" action="{{ url_for('index') }}">
        <input id="question" type="text" name="question" placeholder="Stel je vraag… (bijv. 'Wanneer mag ik ibuprofen gebruiken tijdens zwangerschap?')" autocomplete="off">
        <button type="submit">Vraag</button>
        <a href="{{ url_for('clear') }}"><button type="button" class="danger">Leeg chat</button></a>
      </form>

      <div class="chat">
        {% if not chat %}
          <div class="bot msg">
            <div class="bubble">
              <strong>Welkom!</strong> Stel je geneesmiddelvraag. Ik antwoord op basis van je lokale FAISS-database (apotheek.nl).
            </div>
          </div>
        {% endif %}
        {% for turn in chat %}
          <div class="user msg">
            <div class="bubble">{{ turn.q | e }}</div>
          </div>
          <div class="bot msg">
            <div class="bubble">
              {{ turn.a | replace('\n','<br>') | safe }}
              {% if turn.sources %}
              <div class="sources">
                <div><strong>Bronnen:</strong></div>
                <ul style="margin:6px 0 0 18px; padding:0;">
                {% for s in turn.sources %}
                  <li>
                    [{{ s.id }}]
                    {{ s.place | e }}
                    {% if s.url %}
                      — <a href="{{ s.url }}" target="_blank" rel="noopener">link</a>
                    {% endif %}
                  </li>
                {% endfor %}
                </ul>
              </div>
              {% endif %}
            </div>
          </div>
        {% endfor %}
      </div>
    </div>

    <div class="footer">
      Made with Flask · FAISS · Groq · SentenceTransformers · Dark Navy UI
    </div>
  </div>
</body>
</html>
"""

def _pick_sources_from_answer(hits: List[Tuple[float, Dict]], answer: str):
    used = []
    for i in range(1, len(hits)+1):
        if f"[{i}]" in answer:
            used.append(i)
    if not used:
        used = list(range(1, min(4, len(hits))+1))
    out = []
    for i in used:
        _, m = hits[i-1]
        place = f"{m.get('title','')} > {m.get('section','')}"
        if m.get("subsection"):
            place += f" > {m['subsection']}"
        url = m.get("url") or m.get("source_file") or ""
        out.append({"id": i, "place": place, "url": url})
    return out

@app.route("/", methods=["GET", "POST"])
def index():
    chat = session.get("chat", [])
    if request.method == "POST":
        question = (request.form.get("question") or "").strip()
        if question:
            try:
                # Retrieve
                hits = retrieve(VECTORDB_DIR, question, k=TOP_K)
                if not hits:
                    answer = "Geen context gevonden in de vector database. Bouw eerst de index of pas je vraag aan."
                    sources = []
                else:
                    # Context + LLM
                    ctx_blocks = build_context_blocks(hits)
                    messages   = make_messages(question, ctx_blocks)
                    answer     = groq_chat(messages, model=RAG_MODEL, max_tokens=700, temperature=0.2, stream=False)
                    sources    = _pick_sources_from_answer(hits, answer)
            except Exception as e:
                answer = f"Er ging iets mis: {e}"
                sources = []
            chat.append({"q": question, "a": answer, "sources": sources})
            session["chat"] = chat
        return redirect(url_for("index"))

    return render_template_string(
        HTML,
        chat=chat,
        db_dir=str(VECTORDB_DIR),
        top_k=TOP_K,
        model=RAG_MODEL,
    )

@app.route("/clear")
def clear():
    session.pop("chat", None)
    return redirect(url_for("index"))

@app.route("/healthz")
def healthz():
    return {"ok": True, "db": str(VECTORDB_DIR), "top_k": TOP_K, "model": RAG_MODEL}

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=True)
