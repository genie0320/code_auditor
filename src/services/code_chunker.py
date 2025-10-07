# src/services/code_chunker.py

from .chunking_strategies import ChunkingStrategy  # 전략 인터페이스 임포트
from ..models import FileContext, Chunk


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


# import hashlib
# import tiktoken
# from langchain_text_splitters import (
#     RecursiveCharacterTextSplitter,
#     Language,
# )
# from ..models import FileContext, Chunk


# class CodeChunker:
#     def __init__(self):
#         self.tokenizer = tiktoken.get_encoding("cl100k_base")
#         print("[CodeChunker] Initialized with Tree-sitter support.")

#     # (수정) 파일 확장자에 맞는 언어 Enum을 반환하는 헬퍼 메서드
#     def _get_language_enum(self, extension: str):
#         mapping = {
#             ".py": Language.PYTHON,
#             ".js": Language.JS,
#             ".ts": Language.TS,
#             ".java": Language.JAVA,
#             ".go": Language.GO,
#             ".cpp": Language.CPP,
#             ".md": Language.MARKDOWN,
#             ".html": Language.HTML,
#         }
#         return mapping.get(extension)

#     def chunk_file(self, file_context: FileContext) -> list[Chunk]:
#         print(
#             f"[CodeChunker] Chunking file with Tree-sitter: {file_context.filename}..."
#         )
#         try:
#             with open(file_context.full_path, "r", encoding="utf-8") as f:
#                 content = f.read()
#         except Exception as e:
#             print(f"[CodeChunker] Error reading file {file_context.filename}: {e}")
#             return []

#         language = self._get_language_enum(file_context.extension)
#         if not language:
#             # 지원하지 않는 언어는 기존처럼 라인 기반으로 분할 (Fallback)
#             lines = content.splitlines(True)
#             # ... (이전의 라인 기반 청킹 로직을 여기에 구현) ...
#             return []

#         # (수정) Tree-sitter 기반 분할기 생성
#         splitter = RecursiveCharacterTextSplitter.from_language(
#             language=language, chunk_size=500, chunk_overlap=50
#         )

#         documents = splitter.create_documents([content])

#         chunks = []
#         for doc in documents:
#             chunk_content = doc.page_content
#             # Tree-sitter는 라인 번호 정보를 직접 제공하지 않으므로, 이 부분은 추후 고도화 필요
#             # 우선 청크의 시작 라인만 근사치로 계산
#             start_line = content.count("\n", 0, content.find(chunk_content)) + 1
#             line_count = chunk_content.count("\n")

#             unique_str = f"{str(file_context.full_path)}:{chunk_content}"
#             chunk_id = hashlib.sha256(unique_str.encode("utf-8")).hexdigest()
#             token_count = len(self.tokenizer.encode(chunk_content))

#             chunk = Chunk(
#                 chunk_id=chunk_id,
#                 parent_file_id=file_context.file_id,
#                 job_id=file_context.job_id,
#                 start_line=start_line,
#                 end_line=start_line + line_count,
#                 token_count=token_count,
#                 chunk_content=chunk_content,
#             )
#             chunks.append(chunk)

#         return chunks

# --------------------------------------------------------------------------------------------

# from pathlib import Path
# import config
# import tiktoken  # tiktoken 임포트
# import uuid  # chunk_id 생성을 위해 uuid 임포트
# import hashlib  # 해시 생성을 위한 hashlib 임포트
# from src.models import FileContext, Chunk  # Chunk 모델 임포트


# class CodeChunker:
#     def __init__(self):
#         # config 모듈에서 직접 청킹 설정을 가져와 사용
#         self.chunk_size = config.CHUNK_SIZE_LINES
#         self.overlap = config.CHUNK_OVERLAP_LINES
#         self.tokenizer = tiktoken.get_encoding("cl100k_base")

#         # 2. (개선) 잘못된 설정 값을 초기화 시점에 확인하여 차단
#         if self.overlap >= self.chunk_size:
#             raise ValueError(
#                 f"Overlap({self.overlap}) cannot be larger than or equal to Chunk size({self.chunk_size})."
#             )

#         print(
#             f"[CodeChunker] Initialized. Chunk size: {self.chunk_size}, Overlap: {self.overlap}"
#         )

#     def chunk_file(self, file_context: FileContext) -> list[Chunk]:
#         """
#         파일을 청킹하고 토큰 수 및 상세 메타데이터가 포함된 Chunk 객체 리스트를 반환.
#         """
#         print(f"[CodeChunkar] Chunking file: {file_context.filename}...")
#         try:
#             with open(file_context.full_path, "r", encoding="utf-8") as f:
#                 lines = f.readlines()
#         except Exception as e:
#             print(f"[CodeChunker] Error reading file {file_context.filename}: {e}")
#             return []

#         # 실제 청킹 로직 구현
#         chunks = []
#         i = 0
#         while i < len(lines):
#             start_line_num = i + 1
#             end_line_index = i + self.chunk_size
#             chunk_lines = lines[i:end_line_index]
#             chunk_content = "".join(chunk_lines)

#             # 토큰 수 계산
#             token_count = len(self.tokenizer.encode(chunk_content))

#             # UUID 대신 내용 기반 SHA256 해시로 chunk_id 생성
#             # 청크 내용과 부모 파일 ID를 조합하여 ID를 만들어, 다른 파일의 동일 내용 청크와 구분
#             unique_str = f"{file_context.full_path}:{chunk_content}"
#             chunk_id = hashlib.sha256(unique_str.encode("utf-8")).hexdigest()
#             # Chunk 객체 생성
#             chunk = Chunk(
#                 job_id=file_context.job_id,
#                 parent_file_id=file_context.file_id,
#                 chunk_id=chunk_id,
#                 start_line=start_line_num,
#                 end_line=start_line_num + len(chunk_lines) - 1,
#                 token_count=token_count,
#                 chunk_content=chunk_content,
#             )
#             chunks.append(chunk)
#             i += self.chunk_size - self.overlap

#         return chunks
