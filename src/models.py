# src/models.py
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Chunk:
    """청크의 모든 메타데이터를 담는 객체."""

    chunk_id: str
    parent_file_id: str
    job_id: str
    chunk_method: str
    start_line: int
    end_line: int
    token_count: int
    chunk_content: str


@dataclass
class FileContext:
    """파일 스캔 시점에 생성되는 모든 메타데이터를 담는 객체."""

    # --- 기본 정보 ---
    job_id: str
    file_id: str
    full_path: Path
    directory: Path
    filename: str
    extension: str

    # --- 라인 수 정보 ---
    total_lines: int
    code_lines: int
    comment_lines: int

    # --- 처리 상태 (미래 확장용) ---
    chunks: list[dict] = field(default_factory=list)
    audit_results: list[dict] = field(default_factory=list)
    status: str = ""
