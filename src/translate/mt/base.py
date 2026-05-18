from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Translation:
    source: str
    target: str


class MTWorker(ABC):
    @abstractmethod
    def translate(self, text_ja: str) -> Translation: ...
