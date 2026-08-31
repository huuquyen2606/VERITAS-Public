from abc import ABC, abstractmethod


class Action(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def apply(self, bytez: bytes) -> bytes:
        pass

    def __str__(self):
        return self.name
