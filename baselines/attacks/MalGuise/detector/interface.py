"""
Interface definition for MalGuise Target System.
"""
from dataclasses import dataclass

@dataclass
class DetectorResult:
    malware_prob: float
    is_malware: bool
    cfg_path: str = None
    cfg_source_sha256: str = None

class DetectorInterface:
    def __init__(self, use_probability: bool = True):
        """
        Args:
            use_probability (bool): Whether to use predicted probabilities (w/ prob) or just labels (w/o prob)
        """
        self.use_probability = use_probability

    def predict(self, pe_bytes: bytes, preserve_artifacts: bool = False) -> DetectorResult:
        """
        Evaluate the adversarial PE bytes.
        
        Args:
            pe_bytes (bytes): The raw bytes of the PE file.
            preserve_artifacts (bool): If True, preserves intermediate files (like CFG) and returns their paths.
            
        Returns:
            DetectorResult
        """
        raise NotImplementedError
