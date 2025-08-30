#!/usr/bin/env python3
"""
Utils voor de apotheek.nl scraper:
- Robots.txt check met caching
- RateLimiter
- Fetch met retries/backoff
- Bestandshelpers
"""
from __future__ import annotations
import time, re, json, os
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
import requests
from urllib import robotparser

DEFAULT_UA = os.getenv(
    "SCRAPER_USER_AGENT",
    "apotheek-scraper/1.1 (+contact: you@example.com)"
)

class RateLimiter:
    def __init__(self, min_interval_sec: float = 2.0):
        self.min_interval = float(min_interval_sec)
        self._last = 0.0

    def wait(self):
        now = time.time()
        delta = now - self._last
        if delta < self.min_interval:
            time.sleep(self.min_interval - delta)
        self._last = time.time()

class RobotsCache:
    def __init__(self, session: Optional[requests.Session] = None, timeout: float = 10.0):
        self._cache = {}
        self.session = session or requests.Session()
        self.timeout = timeout

    def _robots_url(self, url: str) -> str:
        p = urlparse(url)
        return f"{p.scheme}://{p.netloc}/robots.txt"

    def allowed(self, url: str, user_agent: str = DEFAULT_UA) -> bool:
        p = urlparse(url)
        key = f"{p.scheme}://{p.netloc}"
        rp = self._cache.get(key)
        if not rp:
            rp = robotparser.RobotFileParser()
            robots_url = self._robots_url(url)
            try:
                resp = self.session.get(robots_url, timeout=self.timeout)
                if resp.status_code >= 400:
                    # Geen robots.txt of niet bereikbaar → val vriendelijk terug op "toegestaan"
                    rp.parse([])
                else:
                    rp.parse(resp.text.splitlines())
            except Exception:
                rp.parse([])
            self._cache[key] = rp
        try:
            return rp.can_fetch(user_agent, url)
        except Exception:
            return True

def fetch_url(url: str, session: Optional[requests.Session] = None,
              rate: Optional[RateLimiter] = None, timeout: float = 30.0,
              max_retries: int = 3, user_agent: str = DEFAULT_UA) -> str:
    sess = session or requests.Session()
    sess.headers.update({"User-Agent": user_agent})
    backoff = 1.5
    attempt = 0
    while True:
        if rate: rate.wait()
        try:
            resp = sess.get(url, timeout=timeout)
            if resp.status_code >= 500:
                raise requests.HTTPError(f"{resp.status_code}")
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                raise
            time.sleep(backoff ** attempt)

def is_url(resource: str) -> bool:
    return bool(re.match(r"^https?://", resource, re.I))

def read_local(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")

def write_json(obj, path: str | Path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def basename_from_resource(resource: str) -> str:
    if is_url(resource):
        path = urlparse(resource).path.rstrip("/")
        name = Path(path).name or "index"
        return name
    return Path(resource).stem

def derive_children_url(adult_url: str) -> str:
    """Heuristisch: voeg '-bij-kinderen/kindertekst' toe aan het laatste padsegment."""
    from urllib.parse import urlparse, urlunparse
    p = urlparse(adult_url)
    segs = [s for s in p.path.split("/") if s]
    if not segs:
        return adult_url  # geen idee → geef terug
    segs[-1] = segs[-1] + "-bij-kinderen"
    segs.append("kindertekst")
    new_path = "/" + "/".join(segs)
    return urlunparse((p.scheme, p.netloc, new_path, "", "", ""))
