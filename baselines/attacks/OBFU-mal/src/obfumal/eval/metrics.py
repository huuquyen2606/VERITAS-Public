from typing import Any, Dict, List, Optional


def summarize_history(
    history: Dict[str, Dict[str, Any]],
    *,
    episode_history: Optional[List[Dict[str, Any]]] = None,
    prepared_total: Optional[int] = None,
    skipped_benign: int = 0,
) -> Dict[str, Any]:
    records = episode_history if episode_history is not None else list(history.values())
    attacked_total = len(records)
    unique_attacked = len(history)
    if prepared_total is None:
        prepared_total = unique_attacked + skipped_benign
    if attacked_total == 0:
        return {
            "total": 0,
            "attacked_total": 0,
            "episodes_started": 0,
            "unique_attacked": unique_attacked,
            "prepared_total": prepared_total,
            "skipped_benign": skipped_benign,
            "evaded": 0,
            "evasion_rate": 0.0,
            "avg_actions": 0.0,
        }

    evaded = sum(1 for h in records if h.get("evaded"))
    avg_actions = sum(len(h.get("actions", [])) for h in records) / attacked_total
    return {
        "total": attacked_total,
        "attacked_total": attacked_total,
        "episodes_started": attacked_total,
        "unique_attacked": unique_attacked,
        "prepared_total": prepared_total,
        "skipped_benign": skipped_benign,
        "evaded": evaded,
        "evasion_rate": evaded / attacked_total,
        "avg_actions": avg_actions,
    }
