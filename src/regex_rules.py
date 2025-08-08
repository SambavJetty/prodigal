import re
from typing import List, Dict, Tuple
from .textnorm import normalize_text_basic, normalize_for_lexicon
from .utils import normalize_speaker

# Profanity stems
PROFANITY_TERMS = [
    "fuck", "shit", "bitch", "asshole", "bastard", "crap", "damn", "hell", "motherf"
]

def _sep_word(w: str) -> str:
    # allow non-word separators between letters (e.g., f***, sh!t)
    return r"".join([re.escape(ch) + r"[\W_]*" for ch in w])

# tolerant regex for typical profanities
_PROFANITY_RE = re.compile(
    "(?i)(?<!\\w)("
    + "|".join([_sep_word(w) for w in ["fuck", "shit", "bitch", "asshole", "bastard"]])
    + r"|crap|damn|hell|motherf[\W_]*\w+"
    + ")(?!\\w)"
)

def _has_profane_norm(text: str) -> bool:
    t = normalize_for_lexicon(text)
    return any(term in t for term in PROFANITY_TERMS)

# Sensitive disclosure (tight keywords only)
_SENSITIVE_RE = re.compile(
    r"(?i)(?<!\w)("
    r"balance|amount\s*due|you\s*owe|outstanding|past\s*due|"
    r"account\s*number|acct\s*(?:number|#)|routing\s*(?:number)?|"
    r"credit\s*card(?:\s*number)?|card\s*number|last\s*four|ending\s*in|cvv"
    r")(?!\w)"
)

# Verification phrases
_VERIF_REQUEST_RE = re.compile(
    r"(?i)\b(verify|verification|confirm|confirmation|for\s+security)\b.*\b("
    r"date\s*of\s*birth|dob|address|social\s*security|ssn"
    r")\b"
)
_VERIF_PROVIDE_RE = re.compile(
    r"(?i)\b("
    r"\d{3}[-\s]?\d{2}[-\s]?\d{4}|"  # SSN
    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},\s*\d{4}|"  # July 15, 1990
    r"\d{1,2}/\d{1,2}/\d{2,4}|"  # 07/15/1990
    r"\d+\s+\w+(?:\s+\w+)*\s+(?:st|street|rd|road|ave|avenue|dr|drive)\.?"
    r")\b"
)

def contains_profanity(text: str) -> bool:
    if not text:
        return False
    return bool(_PROFANITY_RE.search(text)) or _has_profane_norm(text)

def detect_profanity_by_role_regex(utterances: List[Dict]) -> Tuple[bool, bool]:
    agent_flag = False
    borrower_flag = False
    for u in utterances:
        if contains_profanity(u.get("text", "")):
            role = normalize_speaker(u.get("speaker", ""))
            if role == "Agent":
                agent_flag = True
            elif role == "Borrower":
                borrower_flag = True
        if agent_flag and borrower_flag:
            break
    return agent_flag, borrower_flag

def detect_privacy_violation_regex(utterances: List[Dict]) -> bool:
    verified = False
    agent_requested = False
    for u in utterances:
        role = normalize_speaker(u.get("speaker", ""))
        text = normalize_text_basic(u.get("text", ""))

        if role == "Agent":
            if _VERIF_REQUEST_RE.search(text):
                agent_requested = True
            if _SENSITIVE_RE.search(text):
                if not verified:
                    return True
        else:
            if _VERIF_PROVIDE_RE.search(text):
                verified = True
    return False
