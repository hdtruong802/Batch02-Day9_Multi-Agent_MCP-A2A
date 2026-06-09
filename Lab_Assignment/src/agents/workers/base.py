"""Base worker interface for Supervisor-Workers pattern."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class WorkerResult:
    """Kết quả trả về từ một worker."""

    worker_id: str
    worker_name: str
    chunks: list[dict] = field(default_factory=list)
    summary: str = ""
    metadata: dict = field(default_factory=dict)


class BaseWorker(ABC):
    """Worker cơ sở — mỗi worker xử lý một domain cụ thể."""

    worker_id: str = "base"
    worker_name: str = "Base Worker"

    @abstractmethod
    def run(self, query: str, *, top_k: int = 5) -> WorkerResult:
        """Thực thi nhiệm vụ của worker."""
