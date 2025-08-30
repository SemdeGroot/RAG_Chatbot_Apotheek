#!/usr/bin/env python3
"""
Gemeenschappelijke helpers voor het bouwen/raadplegen van een FAISS index.
- Minimale dependencies: numpy, faiss-cpu, sentence-transformers
- Standaard embedding model: intfloat/multilingual-e5-base (geschikt voor NL)
- Cosine similariteit via genormaliseerde embeddings + IndexFlatIP

Let op: gebruik hetzelfde model voor build en query.
"""
from __future__ import annotations
import json, os, re, hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

# Lazy imports (zodat 'query' sneller start en je foutmeldingen netter zijn)
def _require_faiss():
    try:
        import faiss  # type: ignore
        return faiss
    except Exception as e:
        raise RuntimeError("FAISS ontbreekt. Installeer met: pip install faiss-cpu") from e

def _require_st():
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        return SentenceTransformer
    except Exception as e:
        raise RuntimeError("sentence-transformers ontbreekt. Installeer met: pip install sentence-transformers") from e


# ======== Config / defaults ========
DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
DOC_PREFIX = "passage: "
Q_PREFIX = "query: "


# ======== I/O helpers ========
def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def load_clean_json_files(input_dir: Path, pattern: str) -> List[Path]:
    files = sorted(input_dir.glob(pattern))
    return [p for p in files if p.is_file()]


# ======== Chunking ========
_ws = re.compile(r"\s+")

def normalize_space(s: str) -> str:
    return _ws.sub(" ", (s or "").strip())

def _hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()[:16]

def flatten_chunks_from_clean_json(doc: Dict[str, Any], default_url: str = "") -> List[Dict[str, Any]]:
    """
    Zet *_clean.json om naar retrieval-chunks.
    - Elke paragraaf → één chunk
    - Elke list-item → één chunk
    - Context (titel/sectie/subsectie) wordt in 'text' meegenomen
    Metadata:
      text        (context || inhoud)
      raw_text    (alleen inhoud)
      title, section, subsection, block_type, url, source_file
    """
    title = doc.get("title") or ""
    url = doc.get("url") or default_url
    chunks: List[Dict[str, Any]] = []

    def add(text: str, sec: str, sub: str, block_type: str):
        raw = normalize_space(text)
        if not raw:
            return
        ctx = f"Titel: {title} | Sectie: {sec}"
        if sub:
            ctx += f" > {sub}"
        content = ctx + " || " + raw
        chunks.append({
            "text": content,
            "raw_text": raw,
            "title": title,
            "section": sec or "",
            "subsection": sub or None,
            "block_type": block_type,
            "url": url or None,
        })

    for sec in doc.get("sections", []):
        sec_title = sec.get("title") or ""
        for b in sec.get("blocks", []) or []:
            if b.get("type") == "paragraph":
                add(b.get("text", ""), sec_title, "", "paragraph")
            elif b.get("type") == "list":
                for it in b.get("items", []) or []:
                    add(it, sec_title, "", "list_item")
        for sub in sec.get("subsections", []) or []:
            sub_title = sub.get("title") or ""
            for b in sub.get("blocks", []) or []:
                if b.get("type") == "paragraph":
                    add(b.get("text", ""), sec_title, sub_title, "paragraph")
                elif b.get("type") == "list":
                    for it in b.get("items", []) or []:
                        add(it, sec_title, sub_title, "list_item")
    return chunks


# ======== Embeddings / Index ========
def build_embeddings(texts: List[str], model_name: str = DEFAULT_MODEL, batch_size: int = 64) -> np.ndarray:
    SentenceTransformer = _require_st()
    model = SentenceTransformer(model_name)
    to_encode = [DOC_PREFIX + t for t in texts]
    embs = model.encode(to_encode, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    return np.asarray(embs, dtype="float32")

def _index_flat_ip(dim: int):
    faiss = _require_faiss()
    return faiss.IndexFlatIP(dim)

def write_index(outdir: Path, embeddings: np.ndarray, metadata: List[Dict[str, Any]], model_name: str):
    """
    Schrijft:
      - index.faiss      (FAISS index)
      - meta.jsonl       (één JSON per regel met chunk-metadata)
      - config.json      (modelnaam, metric, aantal vectors)
    """
    faiss = _require_faiss()
    outdir.mkdir(parents=True, exist_ok=True)
    dim = int(embeddings.shape[1])

    index = _index_flat_ip(dim)  # cosine via genormaliseerde vectors
    index.add(embeddings)
    faiss.write_index(index, str(outdir / "index.faiss"))

    with (outdir / "meta.jsonl").open("w", encoding="utf-8") as f:
        for m in metadata:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    cfg = {"model_name": model_name, "metric": "cosine", "count": int(embeddings.shape[0])}
    (outdir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

def read_index_and_meta(db_dir: Path):
    faiss = _require_faiss()
    index = faiss.read_index(str(db_dir / "index.faiss"))
    metas: List[Dict[str, Any]] = []
    with (db_dir / "meta.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                metas.append(json.loads(line))
    cfg = json.loads((db_dir / "config.json").read_text(encoding="utf-8"))
    return index, metas, cfg

def embed_query(q: str, model_name: str) -> np.ndarray:
    SentenceTransformer = _require_st()
    model = SentenceTransformer(model_name)
    vec = model.encode([Q_PREFIX + q], normalize_embeddings=True)
    return np.asarray(vec, dtype="float32")

def search(db_dir: Path, query: str, k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
    index, metas, cfg = read_index_and_meta(db_dir)
    q = embed_query(query, cfg.get("model_name", DEFAULT_MODEL))
    scores, idxs = index.search(q, k)
    out: List[Tuple[float, Dict[str, Any]]] = []
    for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
        if idx == -1:
            continue
        out.append((float(score), metas[idx]))
    return out