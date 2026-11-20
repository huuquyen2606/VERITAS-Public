import hashlib
from detector.interface import DetectorInterface, DetectorResult

class SyntheticDetector(DetectorInterface):
    """
    A synthetic non-linear/rugged reward landscape to test MCTS search capability.
    This provides a deterministic oracle without relying on a real detector like MAGIC,
    while ensuring the objective function has local optima to challenge the search.
    """
    def __init__(self, use_probability: bool = True):
        super().__init__(use_probability=use_probability)

    def predict(self, pe_bytes: bytes) -> DetectorResult:
        hasher = hashlib.md5()
        hasher.update(pe_bytes)
        hex_digest = hasher.hexdigest()
        
        # Take chunks of the hash to create pseudo-peaks (local optima)
        val1 = int(hex_digest[0:4], 16) / 65535.0
        val2 = int(hex_digest[4:8], 16) / 65535.0
        val3 = int(hex_digest[8:12], 16) / 65535.0
        
        # Combine them to form a score
        score = (val1 * 0.5) + (val2 * 0.3) + (val3 * 0.2)
        
        # Threshold for malware
        is_malware = score > 0.5
        
        return DetectorResult(
            malware_prob=score,
            is_malware=is_malware
        )
