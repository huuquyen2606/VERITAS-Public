from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union


def _iter_records(
    history: Union[Dict[str, Dict[str, Any]], Sequence[Dict[str, Any]]]
) -> Iterable[Tuple[str, Dict[str, Any]]]:
    if isinstance(history, dict):
        for sha, info in history.items():
            yield sha, info
        return

    for info in history:
        sha = str(info.get("original_sha256") or info.get("sha256") or "")
        yield sha, info


def extract_sequences(history: Union[Dict[str, Dict[str, Any]], Sequence[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    rows = []
    for sha, info in _iter_records(history):
        rows.append(
            {
                "sha256": info.get("original_sha256", sha),
                "evaded": info.get("evaded", False),
                "evaded_sha256": info.get("evaded_sha256"),
                "actions": info.get("actions", []),
                "original_score": info.get("original_score"),
            }
        )
    return rows
