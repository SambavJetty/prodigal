from __future__ import annotations

from typing import List, Dict, Tuple
from functools import lru_cache

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from .textnorm import normalize_text_basic, normalize_for_lexicon
from .regex_rules import PROFANITY_TERMS, contains_profanity

# Constrained vocabularies to keep behavior stable and aligned
_PROF_VOCAB = PROFANITY_TERMS
_SENSITIVE_TOKENS = [
    "balance", "amount", "due", "owe", "outstanding", "past", "routing", "number",
    "credit", "card", "last", "four", "ending", "cvv", "acct", "account"
]
_VERIF_TOKENS = ["verify", "verification", "confirm", "confirmation", "security",
                 "date", "birth", "dob", "address", "social", "ssn",
                 "january","february","march","april","may","june","july",
                 "august","september","october","november","december"]

def _build_vectorizer(vocab_list):
    return CountVectorizer(vocabulary=sorted(set(vocab_list)), lowercase=True)

def _train_logreg(vec: CountVectorizer, texts: List[str], labels: List[int]) -> LogisticRegression:
    X = vec.transform(texts)
    clf = LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=1000)
    clf.fit(X, labels)
    return clf

@lru_cache(maxsize=1)
def get_profanity_model():
    pos = [
        "what the hell",
        "this is crap",
        "you asshole",
        "holy shit",
        "fuck off",
        "bastard",
        "you bitch",
        "motherf xxx",
        "damn, man"
    ]
    neg = [
        "please verify your address",
        "your balance is updated",
        "thank you for your help",
        "how would you like to proceed",
        "no issues from my side",
        "good afternoon",
    ]
    vec = _build_vectorizer(_PROF_VOCAB)
    texts = pos + neg
    labels = [1]*len(pos) + [0]*len(neg)
    clf = _train_logreg(vec, texts, labels)
    return (vec, clf)

@lru_cache(maxsize=1)
def get_sensitive_model():
    pos = [
        "your balance is 200 dollars",
        "you owe 150",
        "the amount due is 50",
        "your account number ends in 1234",
        "credit card number",
        "provide your routing number",
        "card number ending in 5555",
        "last four of the card",
    ]
    neg = [
        "please verify your date of birth",
        "confirm your address for security",
        "thanks for confirming",
        "how can i help you today",
        "we received your payment",
    ]
    vec = _build_vectorizer(_SENSITIVE_TOKENS)
    texts = pos + neg
    labels = [1]*len(pos) + [0]*len(neg)
    clf = _train_logreg(vec, texts, labels)
    return (vec, clf)

@lru_cache(maxsize=1)
def get_verification_model():
    pos = [
        "please verify your date of birth",
        "can you confirm your address for security",
        "what is your social security number",
        "my dob is july 15 1990",
        "it's 123 elm street springfield",
        "my ssn is 123 45 6789",
    ]
    neg = [
        "your balance is 250",
        "we need payment today",
        "thank you for your help",
        "let us proceed with payment",
    ]
    vec = _build_vectorizer(_VERIF_TOKENS)
    texts = pos + neg
    labels = [1]*len(pos) + [0]*len(neg)
    clf = _train_logreg(vec, texts, labels)
    return (vec, clf)

def ml_detect_profanity_by_role(utterances: List[Dict], model) -> Tuple[bool, bool]:
    """
    ML decision is a union (OR) of:
      - lexicon/regex presence (contains_profanity)
      - classifier prediction on constrained vocabulary
    This keeps results aligned with Pattern Matching on most files while allowing
    ML to contribute when trained further.
    """
    vec, clf = model
    agent_flag = False
    borrower_flag = False
    for u in utterances:
        text = u.get("text", "") or ""
        has_lex = contains_profanity(text)
        pred = int(clf.predict(vec.transform([text]))[0])
        is_profane = has_lex or (pred == 1)
        if is_profane:
            sp = str(u.get("speaker", "")).strip().lower()
            if sp == "agent":
                agent_flag = True
            elif sp in ("borrower", "customer"):
                borrower_flag = True
        if agent_flag and borrower_flag:
            break
    return agent_flag, borrower_flag

def ml_detect_privacy_violation(utterances: List[Dict], sens_model, ver_model) -> bool:
    s_vec, s_clf = sens_model
    v_vec, v_clf = ver_model
    verified = False
    for u in utterances:
        speaker = str(u.get("speaker", "")).strip().lower()
        text = normalize_text_basic(u.get("text", "") or "")
        v_pred = int(v_clf.predict(v_vec.transform([text]))[0])
        if v_pred == 1:
            verified = True
            continue
        if speaker == "agent":
            s_pred = int(s_clf.predict(s_vec.transform([text]))[0])
            if s_pred == 1 and not verified:
                return True
    return False
