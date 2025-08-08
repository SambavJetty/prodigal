from typing import List, Dict, Tuple

def _merge_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e:
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))
    return merged

def _interval_length(intervals):
    return sum(e - s for s, e in intervals)

def _pairwise_overlap(intervals_a, intervals_b):
    i = j = 0
    overlap = 0.0
    while i < len(intervals_a) and j < len(intervals_b):
        a_s, a_e = intervals_a[i]
        b_s, b_e = intervals_b[j]
        s = max(a_s, b_s)
        e = min(a_e, b_e)
        if e > s:
            overlap += e - s
        if a_e < b_e:
            i += 1
        else:
            j += 1
    return overlap

def compute_overtalk_and_silence(utterances: List[Dict]) -> Tuple[float, float]:
    """
    Returns (overtalk_percentage, silence_percentage) for a single call.
    - Overtalk: time where Agent and Borrower speak simultaneously / total call duration.
    - Silence: time where nobody speaks / total call duration.
    """
    if not utterances:
        return 0.0, 100.0

    start = min(u["stime"] for u in utterances)
    end = max(u["etime"] for u in utterances)
    total = max(0.000001, end - start)

    agent_intervals = [(u["stime"], u["etime"]) for u in utterances if str(u["speaker"]).lower() == "agent"]
    cust_intervals = [(u["stime"], u["etime"]) for u in utterances if str(u["speaker"]).lower() in ("borrower", "customer")]
    all_intervals = agent_intervals + cust_intervals

    merged_all = _merge_intervals(all_intervals)
    merged_agent = _merge_intervals(agent_intervals)
    merged_cust = _merge_intervals(cust_intervals)

    speaking_time = _interval_length(merged_all)
    silence = max(0.0, total - speaking_time)
    overlap = _pairwise_overlap(merged_agent, merged_cust)

    overtalk_pct = max(0.0, min(100.0, (overlap / total) * 100.0))
    silence_pct = max(0.0, min(100.0, (silence / total) * 100.0))
    return overtalk_pct, silence_pct
