# src/llm_clients/base_client.py
from abc import ABC, abstractmethod


class LLMClient(ABC):
    """LLM API 클라이언트의 추상 베이스 클래스 (Interface)"""

    @abstractmethod
    def process_batch(self, code_chunks: list[dict], system_prompt: str) -> list[dict]:
        """코드 청크 배치를 받아 Audit 결과를 반환."""
        pass
