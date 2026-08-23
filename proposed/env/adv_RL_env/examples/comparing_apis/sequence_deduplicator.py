import time

def deduplicate_sequence(api_sequence, lm=5, k=2):
    """
    Compress redundant API call loops from dynamic analysis traces.

    Args:
        api_sequence: List of API calls (strings or dicts).
        lm: Maximum pattern length to search for.
        k: Maximum consecutive duplicates to retain.

    Returns:
        List of API calls after loop deduplication.
    """
    if not api_sequence:
        return []

    result = list(api_sequence)

    while True:
        prev_len = len(result)

        for p in range(1, lm + 1):
            i = 0
            new_seq = []

            while i < len(result):
                pattern = result[i : i + p]

                if i + 2 * p > len(result):
                    new_seq.extend(result[i:])
                    break

                repeats = 1
                idx = i + p
                while idx + p <= len(result) and result[idx : idx + p] == pattern:
                    repeats += 1
                    idx += p

                if repeats > k:
                    for _ in range(k):
                        new_seq.extend(pattern)
                    i = idx
                else:
                    new_seq.append(result[i])
                    i += 1

            result = new_seq

        if len(result) == prev_len:
            break

    return result