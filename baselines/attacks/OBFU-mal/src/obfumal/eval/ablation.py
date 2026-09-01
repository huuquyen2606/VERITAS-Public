from typing import Any, Dict, List

from obfumal.eval.metrics import summarize_history


def filter_by_actions(history: Dict[str, Dict[str, Any]], allowed_actions: List[str]) -> Dict[str, Dict[str, Any]]:
    filtered: Dict[str, Dict[str, Any]] = {}
    for sha, info in history.items():
        actions = info.get("actions", [])
        if all(a in allowed_actions for a in actions):
            filtered[sha] = info
    return filtered


def ablation_summary(history: Dict[str, Dict[str, Any]], allowed_actions: List[str]) -> Dict[str, Any]:
    filtered = filter_by_actions(history, allowed_actions)
    return summarize_history(filtered)
