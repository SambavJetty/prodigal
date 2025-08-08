import re

_WS_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

def normalize_text_basic(text: str) -> str:
    if not text:
        return ""
    t = text.lower()
    t = _WS_RE.sub(" ", t)
    return t.strip()

def normalize_for_lexicon(text: str) -> str:
    if not text:
        return ""
    t = text.lower()
    t = _NON_ALNUM_RE.sub("", t)
    return t
