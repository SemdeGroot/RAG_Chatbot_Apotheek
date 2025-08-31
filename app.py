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

# Herken HF Spaces (iframe => third-party cookies)
IN_SPACES = bool(os.getenv("SPACE_ID") or os.getenv("HF_SPACE"))
SAMESITE  = "None" if IN_SPACES else "Lax"
SECURE    = True   if IN_SPACES else False

try:
    # Server-side sessies (aanrader)
    from flask_session import Session

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
    Session(app)
    print(f"[session] Filesystem sessions -> {SESSION_DIR}")

except Exception as e:
    # Fallback: cookie-sessies (werkt niet altijd in iframe; zet dan USE_POST_REDIRECT=0)
    app.config.update(
        SECRET_KEY=os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me"),
        SESSION_USE_SIGNER=True,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE=SAMESITE,
        SESSION_COOKIE_SECURE=SECURE,
    )
    print(f"⚠️ Flask-Session niet actief: {e}")

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
      background: var(--bg-tertiary); position: relative;
    }
    input[type="text"]{
      flex:1; background: var(--blue-tertiary);
      border: 2px solid transparent; color: var(--text-primary);
      padding: 12px 14px; border-radius: 10px; outline:none;
      transition: border .2s ease, box-shadow .2s ease, transform .1s ease;
    }
    input[type="text"]::placeholder{ color:#91a1c5; }
    input[type="text"]:focus{
      border-color: var(--blue-primary);
      box-shadow: 0 0 0 3px rgba(29,78,216,0.18);
      transform: translateY(-1px);
    }
    input[type="text"]:disabled{
      opacity: 0.6; cursor: not-allowed;
    }
    button{
      appearance:none; border:none; border-radius:10px;
      padding: 12px 16px; font-weight:600; cursor:pointer;
      transition: all .2s ease; color:#fff; position: relative; overflow: hidden;
    }
    .btn-primary{ 
      background: linear-gradient(135deg, var(--blue-primary), var(--blue-secondary)); 
      box-shadow: var(--shadow-sm); 
    }
    .btn-primary:hover:not(:disabled){ 
      transform: translateY(-1px); 
      box-shadow: var(--shadow-md); 
    }
    .btn-primary:disabled{
      opacity: 0.7; cursor: not-allowed; transform: none;
    }
    .btn-danger{ background: linear-gradient(180deg, #d94848, #b33636); }

    /* Loading states */
    .loading-overlay {
      position: absolute;
      top: 0; left: 0; right: 0; bottom: 0;
      background: rgba(15, 15, 23, 0.9);
      backdrop-filter: blur(4px);
      display: none;
      align-items: center;
      justify-content: center;
      z-index: 10;
      border-radius: 10px;
    }
    .loading-overlay.active {
      display: flex;
    }
    .spinner {
      width: 24px; height: 24px;
      border: 3px solid rgba(29, 78, 216, 0.2);
      border-top: 3px solid var(--blue-primary);
      border-radius: 50%;
      animation: spin 1s linear infinite;
    }
    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }

    /* Chat */
    .chat{
      display:flex; flex-direction:column; gap:12px;
      max-height: calc(100vh - 360px); overflow:auto; padding: 12px 12px 16px;
      background: var(--blue-soft); border-top:1px solid var(--border-primary);
    }
    .msg{ 
      display:flex; gap:10px; 
      opacity: 0;
      animation: fadeInUp 0.4s ease forwards;
    }
    @keyframes fadeInUp {
      from {
        opacity: 0;
        transform: translateY(20px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }
    .bubble{
      padding:12px 14px; border-radius: 12px; line-height:1.6;
      border:1px solid rgba(255,255,255,0.08); box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
      background: var(--bg-secondary);
      transition: all 0.2s ease;
    }
    .bubble:hover {
      transform: translateY(-1px);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.06), 0 4px 12px rgba(0,0,0,.3);
    }
    .user .bubble{ background: #0f1b35; border-color: rgba(148,176,255,0.18); }
    .bot  .bubble{ background: #0b0f1a; border-color: rgba(148,176,255,0.14); }

    /* Typing indicator */
    .typing-indicator {
      display: none;
      padding: 12px 14px;
      background: #0b0f1a;
      border: 1px solid rgba(148,176,255,0.14);
      border-radius: 12px;
      margin: 12px 0;
    }
    .typing-indicator.active {
      display: block;
      animation: fadeInUp 0.4s ease forwards;
    }
    .typing-dots {
      display: flex;
      gap: 4px;
      align-items: center;
    }
    .typing-dots span {
      width: 8px;
      height: 8px;
      background: var(--blue-primary);
      border-radius: 50%;
      animation: typingDot 1.4s infinite ease-in-out;
    }
    .typing-dots span:nth-child(1) { animation-delay: 0s; }
    .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
    .typing-dots span:nth-child(3) { animation-delay: 0.4s; }
    @keyframes typingDot {
      0%, 60%, 100% {
        transform: scale(0.8);
        opacity: 0.5;
      }
      30% {
        transform: scale(1.2);
        opacity: 1;
      }
    }

    .sources{
      margin-top:8px; font-size: 13px; color: var(--text-muted);
      border-top:1px dashed rgba(148,176,255,0.18); padding-top:8px;
    }
    .footer{
      margin-top:14px; font-size:12px; color:#91a1c5; text-align:center;
    }

    a{ color: var(--text-accent); text-decoration: none; }
    a:hover{ text-decoration: underline; }

    /* Verbeterde button states */
    .btn-primary::before {
      content: '';
      position: absolute;
      top: 50%;
      left: 50%;
      width: 0;
      height: 0;
      background: rgba(255, 255, 255, 0.2);
      border-radius: 50%;
      transform: translate(-50%, -50%);
      transition: width 0.3s ease, height 0.3s ease;
    }
    .btn-primary:active::before {
      width: 100px;
      height: 100px;
    }

    /* Status indicator */
    .status-indicator {
      position: fixed;
      top: 20px;
      right: 20px;
      padding: 8px 12px;
      background: var(--success-bg);
      border: 1px solid var(--success);
      border-radius: 8px;
      color: var(--success);
      font-size: 12px;
      font-weight: 500;
      opacity: 0;
      transform: translateY(-20px);
      transition: all 0.3s ease;
      z-index: 1000;
    }
    .status-indicator.show {
      opacity: 1;
      transform: translateY(0);
    }
    .status-indicator.processing {
      background: var(--info-bg);
      border-color: var(--info);
      color: var(--info);
    }
    .status-indicator.error {
      background: var(--error-bg);
      border-color: var(--error);
      color: var(--error);
    }

    @media (max-width: 768px){
      main{ padding:0 1rem; margin:1rem auto; }
      .chat{ max-height: unset; }
      form{ flex-direction:column; }
      button{ width:100%; }
      .status-indicator {
        top: 10px;
        right: 10px;
        font-size: 11px;
      }
    }

    @media (prefers-reduced-motion: reduce){
      *{ animation-duration:.01ms !important; transition-duration:.01ms !important; }
      body::before{ animation:none; }
      .msg { animation: none; opacity: 1; }
      .typing-dots span { animation: none; }
    }
  </style>

<script>
  let isProcessing = false;

  function showStatus(message, type = 'success') {
    const indicator = document.getElementById('status-indicator');
    if (!indicator) return;
    indicator.textContent = message;
    indicator.className = `status-indicator ${type} show`;
    setTimeout(() => indicator.classList.remove('show'), 3000);
  }

  function scrollToBottom() {
    const chat = document.querySelector('.chat');
    if (chat) setTimeout(() => (chat.scrollTop = chat.scrollHeight), 50);
  }

  function setFormState(loading) {
    isProcessing = loading;
    const input = document.querySelector('#question');
    const submitBtn = document.querySelector('.btn-primary');
    const overlay = document.querySelector('.loading-overlay');
    if (submitBtn) {
      submitBtn.disabled = loading;
      submitBtn.textContent = loading ? 'Bezig...' : 'Vraag';
    }
    // LET OP: input pas na body-build disablen (zie handler)
    if (overlay) overlay.classList.toggle('active', loading);
  }

  window.addEventListener('load', () => {
    const form = document.querySelector('form');
    const input = document.querySelector('#question');
    const chatBox = document.querySelector('.chat');
    const typing = document.getElementById('typing-indicator');

    if (input) input.focus();
    scrollToBottom();

    if (!form || !input || !chatBox) return;

    form.addEventListener('submit', async (e) => {
      e.preventDefault();

      const q = input.value.trim();
      if (!q || isProcessing) return;

      // 1) Body opbouwen vóórdat we velden disablen
      //    (disabled inputs worden niet meegestuurd!)
      const body = new URLSearchParams();
      body.set('question', q);

      // 2) Nu pas UI op "loading"
      setFormState(true);
      input.disabled = true;
      if (typing) typing.classList.add('active');

      try {
        const resp = await fetch(form.action, {
          method: 'POST',
          headers: { 'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
                     'Accept': 'text/html' },
          body
        });
        if (!resp.ok) throw new Error(resp.status + ' ' + resp.statusText);

        const html = await resp.text();
        const doc = new DOMParser().parseFromString(html, 'text/html');
        const newChat = doc.querySelector('.chat');
        if (newChat) {
          chatBox.innerHTML = newChat.innerHTML;
          input.value = '';
          showStatus('Antwoord ontvangen', 'success');
        } else {
          showStatus('Kon chatweergave niet bijwerken', 'error');
        }
      } catch (err) {
        console.error(err);
        showStatus('Fout: ' + (err?.message || err), 'error');
      } finally {
        if (typing) typing.classList.remove('active');
        input.disabled = false;
        setFormState(false);
        input.focus();
        scrollToBottom();
      }
    });

    document.addEventListener('keydown', (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        form.dispatchEvent(new Event('submit', { cancelable: true }));
      }
    });
  });
</script>

</head>
<body>
  <!-- Status indicator -->
  <div id="status-indicator" class="status-indicator"></div>

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
          
          <!-- Loading overlay voor form -->
          <div class="loading-overlay">
            <div class="spinner"></div>
          </div>
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

          <!-- Typing indicator -->
          <div id="typing-indicator" class="typing-indicator">
            <div class="typing-dots">
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
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