import os

def normalize_speaker(s: str) -> str:
    s_lower = str(s).strip().lower()
    if s_lower in ("agent", "rep", "collector"):
        return "Agent"
    if s_lower in ("borrower", "customer", "consumer", "client"):
        return "Borrower"
    return str(s).strip().title()

def extract_call_id_from_name(name: str) -> str:
    base = os.path.basename(name)
    for ext in (".yaml", ".yml", ".json"):
        if base.lower().endswith(ext):
            return base[: -len(ext)]
    return base
