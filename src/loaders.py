import json
from typing import List, Dict, Any
import yaml

from .utils import normalize_speaker

def _validate_and_normalize(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in records:
        if not all(k in r for k in ("speaker", "text", "stime", "etime")):
            continue
        try:
            speaker = normalize_speaker(str(r["speaker"]))
            text = str(r["text"])
            stime = float(r["stime"])
            etime = float(r["etime"])
            if etime < stime:
                stime, etime = etime, stime
            out.append({"speaker": speaker, "text": text, "stime": stime, "etime": etime})
        except Exception:
            continue
    out.sort(key=lambda x: (x["stime"], x["etime"]))
    return out

def load_calls_from_file(f, name: str) -> List[Dict[str, Any]]:
    content = f.read()
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="ignore")
    if name.lower().endswith((".yaml", ".yml")):
        data = yaml.safe_load(content)
    elif name.lower().endswith(".json"):
        data = json.loads(content)
    else:
        try:
            data = json.loads(content)
        except Exception:
            data = yaml.safe_load(content)
    records = data
    if not isinstance(records, list):
        return []
    return _validate_and_normalize(records)

