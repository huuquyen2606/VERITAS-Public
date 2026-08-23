from __future__ import annotations

from abc import ABC, abstractmethod


class Action(ABC):
    """Path-based action interface for external PE transformation tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def apply(self, file_path: str) -> str:
        """Apply the action to file_path and return the new file path."""
        pass

    def __str__(self) -> str:
        return self.name
