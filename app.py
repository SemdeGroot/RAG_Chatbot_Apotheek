#!/usr/bin/env python3
"""
Flask RAG chatbot (GROQ) – moderne dark UI (zonder logo)

- Leest FAISS DB uit VECTORDB_DIR (default: data/vectordb)
- Leest GROQ_API_KEY uit .env of omgeving
- Server-side sessies via Flask-Session (filesystem, tempdir) met nette fallback
- Chat-history capped om oversized cookies te voorkomen als fallback actief is

Run:
  pip install flask Flask-Session groq sentence-transformers faiss-cpu
  python app.py
"""

from __future__ import annotations
import os, re, sys, secrets, tempfile
from pathlib import Path
from typing import List, Dict, Tuple
from flask import Flask, request, render_template_string, session, redirect, url_for
from urllib.parse import urlparse, urlunparse
from flask_session import Session

# ----------------------
# .env loader (lichtgewicht)
# ----------------------
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

# Zorg dat projectroot importeerbaar blijft, ook als je vanuit een andere map start
sys.path.append(str(Path(__file__).parent.resolve()))

# ----------------------
# Externe modules (RAG)
# ----------------------
from augmented_generation.rag_chat import retrieve, build_context_blocks, make_messages
from augmented_generation.providers.groq_client import chat as groq_chat

# ----------------------
# Config
# ----------------------
VECTORDB_DIR = Path(os.getenv("VECTORDB_DIR", "data/vectordb"))
VECTORDB_DIR_STR = str(VECTORDB_DIR)
RAG_MODEL = os.getenv("RAG_MODEL", "llama-3.3-70b-versatile")
TOP_K = int(os.getenv("RAG_TOPK", "5"))

# ----------------------
# Flask app + Session
# ----------------------
app = Flask(__name__)

# Probeer server-side sessies; val terug op cookie-sessies als Flask-Session ontbreekt

# Herken of we op Hugging Face Spaces draaien
IN_SPACES = bool(os.getenv("SPACE_ID") or os.getenv("HF_SPACE"))

# Session-cookie geschikt maken voor iframe (HF): None + Secure
SAMESITE = "None" if IN_SPACES else "Lax"
SECURE   = True   if IN_SPACES else False

# Server-side sessies in tempdir (werkt op HF)
SESSION_DIR = Path(tempfile.gettempdir()) / "rag_apotheek_sessions"
SESSION_DIR.mkdir(parents=True, exist_ok=True)

app.config.update(
    SECRET_KEY=os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me"),
    SESSION_TYPE="filesystem",
    SESSION_PERMANENT=False,
    SESSION_FILE_DIR=str(SESSION_DIR),
    SESSION_USE_SIGNER=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE=SAMESITE,
    SESSION_COOKIE_SECURE=SECURE,
)

Session(app)  # <— DIT activeert Flask-Session echt

# ----------------------
# Frontend (moderne dark UI – geïnspireerd op je voorbeeld, zonder logo)
# ----------------------
HTML = r"""<!doctype html>
<html lang="nl">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Apotheek.nl Chatbot</title>

  <!-- Inter font -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">

  <style>
    :root{
      /* Donkere basis + blauwe accenten */
      --bg-primary:#0f0f17;
      --bg-secondary:#1a1a26;
      --bg-tertiary:#252532;
      --bg-accent:#2d2d3f;

      --blue-primary:#1d4ed8;
      --blue-secondary:#1e3a8a;
      --blue-tertiary:#0f172a;
      --blue-soft:#1e293b;
      --blue-glow:rgba(29,78,216,0.15);

      --text-primary:#f8fafc;
      --text-secondary:#cbd5e1;
      --text-muted:#94a3b8;
      --text-accent:#e0e7ff;

      --success:#10b981;
      --success-bg:rgba(16,185,129,0.1);
      --error:#ef4444;
      --error-bg:rgba(239,68,68,0.1);
      --info:#1d4ed8;
      --info-bg:rgba(29,78,216,0.1);

      --border-primary:#374151;
      --shadow-sm:0 1px 3px rgba(0,0,0,.3);
      --shadow-md:0 4px 12px rgba(0,0,0,.4);
      --shadow-lg:0 10px 40px rgba(0,0,0,.6);

      --radius-sm:8px; --radius-md:12px; --radius-lg:16px; --radius-xl:20px;
    }
    *{box-sizing:border-box; margin:0; padding:0}
    body{
      font-family:'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
      color:var(--text-primary);
      min-height:100vh; line-height:1.6; overflow-x:hidden;
    }
    body::before{
      content:''; position:fixed; inset:0; z-index:-1;
      background:
        radial-gradient(circle at 20% 20%, rgba(29,78,216,0.12) 0%, transparent 50%),
        radial-gradient(circle at 80% 80%, rgba(29,78,216,0.08) 0%, transparent 50%);
      animation: float 20s ease-in-out infinite;
    }
    @keyframes float{0%,100%{transform:translateY(0) rotate(0)}50%{transform:translateY(-10px) rotate(1deg)}}

    header{
      background: rgba(26,26,38,.95);
      backdrop-filter: blur(12px);
      border-bottom:1px solid var(--border-primary);
      padding:1rem 2rem; position:sticky; top:0; z-index:100; box-shadow: var(--shadow-md);
    }
    .header-content{
      max-width:1200px; margin:0 auto; display:flex; align-items:center; gap:12px;
    }
    .brand-dot{
      width:12px; height:12px; border-radius:50%;
      background: linear-gradient(135deg, var(--blue-primary), var(--success));
      box-shadow: 0 0 18px var(--blue-glow);
    }
    header h1{
      font-size:1.35rem; font-weight:600;
      background: linear-gradient(135deg, var(--text-primary) 0%, var(--blue-primary) 100%);
      -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
      letter-spacing:-.02em;
    }

    main{ max-width:1200px; margin:2rem auto; padding:0 2rem; }

    .card{
      background: var(--bg-tertiary);
      border:1px solid var(--border-primary);
      border-radius: var(--radius-xl);
      box-shadow: var(--shadow-lg); overflow:hidden; position:relative;
    }
    .card::before{ content:''; position:absolute; top:0; left:0; right:0; height:3px;
      background: linear-gradient(90deg, var(--blue-primary), var(--blue-secondary)); }

    .head{ background: linear-gradient(135deg, var(--bg-accent) 0%, var(--bg-tertiary) 100%);
      padding:1.1rem 1.5rem; border-bottom:1px solid var(--border-primary); }
    .head h2{ font-size:1.05rem; font-weight:600; color:var(--text-primary); }

    .content{ padding:1rem 1.25rem 1.25rem; }

    /* Meta bar */
    .meta{
      display:flex; flex-wrap:wrap; gap:8px; align-items:center;
      color:var(--text-muted); font-size:.9rem; margin-bottom:.75rem;
    }
    .badge{
      display:inline-block; padding:2px 8px; border-radius:20px;
      background: #0f1b35; color: #a8b8e8; border:1px solid rgba(148,176,255,0.2);
      font-size:12px;
    }

    /* Form */
    form{
      display:flex; gap:10px; align-items: stretch;
      padding: 12px; border-bottom: 1px solid var(--border-primary);
      background: var(--bg-tertiary);
    }
    input[type="text"]{
      flex:1; background: var(--blue-tertiary);
      border: 2px solid transparent; color: var(--text-primary);
      padding: 12px 14px; border-radius: 10px; outline:none;
      transition: border .2s ease, box-shadow .2s ease;
    }
    input[type="text"]::placeholder{ color:#91a1c5; }
    input[type="text"]:focus{
      border-color: var(--blue-primary);
      box-shadow: 0 0 0 3px rgba(29,78,216,0.18);
      transform: translateY(-1px);
    }
    button{
      appearance:none; border:none; border-radius:10px;
      padding: 12px 16px; font-weight:600; cursor:pointer;
      transition: all .2s ease; color:#fff;
    }
    .btn-primary{ background: linear-gradient(135deg, var(--blue-primary), var(--blue-secondary)); box-shadow: var(--shadow-sm); }
    .btn-primary:hover{ transform: translateY(-1px); box-shadow: var(--shadow-md); }
    .btn-danger{ background: linear-gradient(180deg, #d94848, #b33636); }

    /* Chat */
    .chat{
      display:flex; flex-direction:column; gap:12px;
      max-height: calc(100vh - 360px); overflow:auto; padding: 12px 12px 16px;
      background: var(--blue-soft); border-top:1px solid var(--border-primary);
    }
    .msg{ display:flex; gap:10px; }
    .bubble{
      padding:12px 14px; border-radius: 12px; line-height:1.6;
      border:1px solid rgba(255,255,255,0.08); box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
      background: var(--bg-secondary);
    }
    .user .bubble{ background: #0f1b35; border-color: rgba(148,176,255,0.18); }
    .bot  .bubble{ background: #0b0f1a; border-color: rgba(148,176,255,0.14); }
    .sources{
      margin-top:8px; font-size: 13px; color: var(--text-muted);
      border-top:1px dashed rgba(148,176,255,0.18); padding-top:8px;
    }
    .footer{
      margin-top:14px; font-size:12px; color:#91a1c5; text-align:center;
    }

    a{ color: var(--text-accent); text-decoration: none; }
    a:hover{ text-decoration: underline; }

    @media (max-width: 768px){
      main{ padding:0 1rem; margin:1rem auto; }
      .chat{ max-height: unset; }
      form{ flex-direction:column; }
      button{ width:100%; }
    }

    @media (prefers-reduced-motion: reduce){
      *{ animation-duration:.01ms !important; transition-duration:.01ms !important; }
      body::before{ animation:none; }
    }
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
  <header>
    <div class="header-content">
      <div class="brand-dot" aria-hidden="true"></div>
      <h1>Apotheek.nl Chatbot</h1>
    </div>
  </header>

  <main>
    <section class="card">
      <div class="head">
        <h2>Vraag en antwoord op basis van Apotheek.nl teksten</h2>
      </div>

      <div class="content">

        <form method="post" action="{{ url_for('index') }}">
          <input id="question" type="text" name="question"
                 placeholder="Stel je vraag… (bijv. 'Wat zijn mogelijke bijwerkingen van metoprolol?')"
                 autocomplete="off">
          <button class="btn-primary" type="submit">Vraag</button>
          <a href="{{ url_for('clear') }}"><button class="btn-danger" type="button">Leeg chat</button></a>
        </form>

        <div class="chat">
          {% if not chat %}
            <div class="bot msg">
              <div class="bubble">
                <strong>Welkom!</strong> Stel je geneesmiddelvraag. Ik weet op dit moment alles over paracetamol, ibuprofen, metoprolol, candesartan, amitriptyline, venlafaxine, metformine, omeprazol, amlodipine, atorvastatine, amoxicilline en salbutamol.
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
    </section>

  </main>
</body>
</html>
"""

# ----------------------
# Helpers
# ----------------------
def _canon_url(url: str) -> str | None:
    if not url or not isinstance(url, str):
        return None
    try:
        p = urlparse(url)
        if not p.scheme or not p.netloc:
            return None
        path = (p.path or "/").rstrip("/") or "/"
        return urlunparse((p.scheme.lower(), p.netloc.lower(), path, "", "", ""))
    except Exception:
        return None

def build_sources_from_hits(hits: List[Tuple[float, Dict]]) -> List[Dict]:
    """
    Bouw een unieke bronnenlijst op basis van de top-k hits.
    Dedupliceer op canonieke URL; valt terug op (title, section, subsection) als URL ontbreekt.
    Nummer de zichtbare bronnen 1..N (onafhankelijk van hit-index).
    """
    seen = set()
    sources = []
    next_id = 1

    for _, m in hits:
        url = m.get("url") or m.get("source_file") or ""
        cu = _canon_url(url)
        key = cu or ( (m.get("title","") or "").strip(),
                      (m.get("section","") or "").strip(),
                      (m.get("subsection") or "").strip() )
        if key in seen:
            continue
        seen.add(key)

        title = (m.get("title","") or "").strip()
        section = (m.get("section","") or "").strip()
        subsection = (m.get("subsection") or "").strip()
        place = f"{title} > {section}" + (f" > {subsection}" if subsection else "")

        sources.append({"id": next_id, "place": place, "url": cu or url})
        next_id += 1

    return sources

# ===== Routes & settings =====
# Render direct na POST (aan), of gebruik PRG met redirect (uit)
USE_POST_REDIRECT = os.getenv("USE_POST_REDIRECT", "0") == "1"
CHAT_HISTORY_MAX  = int(os.getenv("CHAT_HISTORY_MAX", "20"))

@app.route("/", methods=["GET", "POST"])
def index():
    chat = session.get("chat", [])

    if request.method == "POST":
        question = (request.form.get("question") or "").strip()
        if question:
            try:
                # 1) Retrieve
                hits = retrieve(VECTORDB_DIR, question, k=TOP_K)
                if not hits:
                    answer  = "Geen context gevonden in de vector database. Bouw eerst de index of pas je vraag aan."
                    sources = []
                else:
                    # 2) Context + LLM
                    ctx_blocks = build_context_blocks(hits)
                    messages   = make_messages(question, ctx_blocks)
                    answer     = groq_chat(messages, model=RAG_MODEL, max_tokens=700, temperature=0.2, stream=False)

                    # 3) Geen 'Bronnen:'-regels in de tekst (we tonen eigen bronnenlijst)
                    answer = re.sub(r"(?im)^\s*bronnen\s*:.*$", "", answer).strip()

                    # 4) Bronnenlijst: unieke URLs
                    sources = build_sources_from_hits(hits)

            except Exception as e:
                answer  = f"Er ging iets mis: {e}"
                sources = []

            # 5) Chat bijwerken + cap
            chat.append({"q": question, "a": answer, "sources": sources})
            chat = chat[-CHAT_HISTORY_MAX:]
            session["chat"] = chat

        if USE_POST_REDIRECT:
            return redirect(url_for("index"))

    # GET of direct render na POST
    return render_template_string(HTML, chat=chat)

@app.route("/clear", methods=["GET"])
def clear():
    session["chat"] = []
    if USE_POST_REDIRECT:
        return redirect(url_for("index"))
    return render_template_string(HTML, chat=[])

@app.route("/healthz")
def healthz():
    return {
        "ok": True,
        "db_dir": str(VECTORDB_DIR),
        "has_index": (VECTORDB_DIR / "index.faiss").exists(),
        "top_k": TOP_K,
        "model": RAG_MODEL,
    }

# ----------------------
# Main
# ----------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=True)