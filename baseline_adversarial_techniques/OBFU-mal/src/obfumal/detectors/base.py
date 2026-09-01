from abc import ABC, abstractmethod
from typing import Any, Dict


class Detector(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def predict(self, bytez: bytes) -> Dict[str, Any]:
        pass
