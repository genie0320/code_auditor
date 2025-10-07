# src/services/code_chunker.py

from src.models import FileContext, Chunk
from services.chunking_strategies import ChunkingStrategy  # 전략 인터페이스 임포트


# Architectural Pattern: Strategy Pattern 적용으로 할일이 대폭 삭감됨.
class CodeChunker:
    """
    주입된 청킹 '전략'을 실행하는 Context 클래스.
    """

    def __init__(self, strategy: ChunkingStrategy):
        self.strategy = strategy
        print(f"[CodeChunker] Initialized with strategy: {strategy.__class__.__name__}")

    def chunk_file(self, file_context: FileContext) -> list[Chunk]:
        """주입된 전략을 사용하여 파일을 청킹합니다."""
        return self.strategy.chunk_file(file_context)
