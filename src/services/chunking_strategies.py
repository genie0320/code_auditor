# src/services/chunking_strategies.py
import hashlib
import tiktoken
from abc import ABC, abstractmethod
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

from src.models import FileContext, Chunk


class ChunkingStrategy(ABC):
    """모든 청킹 전략이 따라야 하는 인터페이스(추상 클래스)"""

    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    @abstractmethod
    def chunk_file(self, file_context: FileContext) -> list[Chunk]:
        pass

    def _create_chunk_from_content(
        self, content: str, file_context: FileContext, start_line: int
    ) -> Chunk:
        """청크 내용으로부터 공통 Chunk 객체를 생성하는 헬퍼 메서드"""
        line_count = content.count("\n")
        unique_str = f"{str(file_context.full_path)}:{content}"
        chunk_id = hashlib.sha256(unique_str.encode("utf-8")).hexdigest()
        token_count = len(self.tokenizer.encode(content))

        return Chunk(
            chunk_id=chunk_id,
            parent_file_id=file_context.file_id,
            job_id=file_context.job_id,
            chunk_method=file_context.chunk_method,
            start_line=start_line,
            end_line=start_line + line_count,
            token_count=token_count,
            chunk_content=content,
        )


class LineChunkingStrategy(ChunkingStrategy):
    """(전략 1) 단순 라인 기반 청킹 전략"""

    def __init__(self, chunk_size: int, overlap: int):
        super().__init__()
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_file(self, file_context: FileContext) -> list[Chunk]:
        """_summary_
        파일을 청킹하고 토큰 수 및 상세 메타데이터가 포함된 Chunk 객체 리스트를 반환.
        파일을 읽고, lines 리스트로 만든 다음, for 루프를 돌며 청크를 생성
        """
        print(f"[CodeChunkar] Chunking file: {file_context.filename}...")
        try:
            with open(file_context.full_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            print(f"[CodeChunker] Error reading file {file_context.filename}: {e}")
            return []

        # 실제 청킹 로직 구현
        chunks = []
        i = 0
        while i < len(lines):
            start_line_num = i + 1
            end_line_index = i + self.chunk_size
            chunk_lines = lines[i:end_line_index]
            chunk_content = "".join(chunk_lines)

            # 토큰 수 계산
            token_count = len(self.tokenizer.encode(chunk_content))

            # UUID 대신 내용 기반 SHA256 해시로 chunk_id 생성
            # 청크 내용과 부모 파일 ID를 조합하여 ID를 만들어, 다른 파일의 동일 내용 청크와 구분
            unique_str = f"{file_context.full_path}:{chunk_content}"
            chunk_id = hashlib.sha256(unique_str.encode("utf-8")).hexdigest()
            # Chunk 객체 생성
            chunk = Chunk(
                job_id=file_context.job_id,
                chunk_method="line",
                parent_file_id=file_context.file_id,
                chunk_id=chunk_id,
                start_line=start_line_num,
                end_line=start_line_num + len(chunk_lines) - 1,
                token_count=token_count,
                chunk_content=chunk_content,
            )
            chunks.append(chunk)

            i += self.chunk_size - self.overlap

        return chunks


class TreeSitterChunkingStrategy(ChunkingStrategy):
    """(전략 2) Tree-sitter 기반 청킹 전략"""

    def __init__(self, chunk_size: int, overlap: int):
        super().__init__()
        self.chunk_size = chunk_size
        self.overlap = overlap

    def _get_language_enum(self, extension: str):
        """# (수정) 파일 확장자에 맞는 언어 Enum을 반환하는 헬퍼 메서드"""
        mapping = {
            ".py": Language.PYTHON,
            ".js": Language.JS,
            ".ts": Language.TS,
            ".java": Language.JAVA,
            ".go": Language.GO,
            ".cpp": Language.CPP,
            ".md": Language.MARKDOWN,
            ".html": Language.HTML,
        }
        return mapping.get(extension)

    def chunk_file(self, file_context: FileContext) -> list[Chunk]:
        print(
            f"[CodeChunker] Chunking file with Tree-sitter: {file_context.filename}..."
        )
        try:
            with open(file_context.full_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            print(f"[CodeChunker] Error reading file {file_context.filename}: {e}")
            return []

        language = self._get_language_enum(file_context.extension)
        if not language:
            # 지원하지 않는 언어는 기존처럼 라인 기반으로 분할 (Fallback)
            lines = content.splitlines(True)
            # ... (이전의 라인 기반 청킹 로직을 여기에 구현) ...
            return []

        # (수정) Tree-sitter 기반 분할기 생성
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=language, chunk_size=500, chunk_overlap=50
        )

        documents = splitter.create_documents([content])

        chunks = []
        for doc in documents:
            chunk_content = doc.page_content
            # TODO: Tree-sitter는 라인 번호 정보를 직접 제공하지 않으므로, 이 부분은 추후 고도화 필요
            # 우선 청크의 시작 라인만 근사치로 계산
            start_line = content.count("\n", 0, content.find(chunk_content)) + 1
            line_count = chunk_content.count("\n")

            unique_str = f"{str(file_context.full_path)}:{chunk_content}"
            chunk_id = hashlib.sha256(unique_str.encode("utf-8")).hexdigest()
            token_count = len(self.tokenizer.encode(chunk_content))

            chunk = Chunk(
                job_id=file_context.job_id,
                parent_file_id=file_context.file_id,
                chunk_id=chunk_id,
                chunk_method="treesitter",
                start_line=start_line,
                end_line=start_line + line_count,
                token_count=token_count,
                chunk_content=chunk_content,
            )
            chunks.append(chunk)

        return chunks
