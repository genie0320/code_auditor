# config.py

import os
from pathlib import Path
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# --- 경로 설정 ---
BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "reports"
ARTIFACTS_DIR = BASE_DIR / "_artifacts"

# --- 청킹(Chunking) 설정 ---
CHUNKING_STRATEGY = "treesitter"  # "line" 또는 "treesitter" 선택 가능
# 라인 기반 청킹 설정
LINE_CHUNK_SIZE = 200
LINE_CHUNK_OVERLAP = 20
# Tree-sitter 기반 청킹 설정
TS_CHUNK_SIZE = 500
TS_CHUNK_OVERLAP = 50

# --- Embedding 설정 ---
EMBEDDING_PROVIDER = "local"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
# EMBEDDING_PROVIDER = "openai"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
# ----------------------------------------

# --- Vector DB 설정 ---
CHROMA_DB_PATH = str(BASE_DIR / "_chroma_db")
SIMILARITY_THRESHOLD = 0.9  # 유사도 검색 임계값

# --- LLM 클라이언트 설정 ---
SELECTED_LLM_CLIENT = "openai"
# 실제 키는 .env 파일에서 로드할 예정
LLM_API_KEYS = {
    "openai": os.getenv("OPENAI_API_KEY"),
    # "gemini": os.getenv("GEMINI_API_KEY"),  # (향후 확장용)
    # "claude": os.getenv("CLAUDE_API_KEY"),  # (향후 확장용)
}
LLM_DEFAULT_MODEL = "gpt-5-nano"
LLM_MAX_TOKENS = 10000

# --- 파일 스캔 설정 ---
TARGET_ROOT_DIR = "_source_code_test"

# 분석에서 제외할 폴더, 확장자, 파일 이름을 명확히 분리
EXCLUDED_FOLDERS = [
    ".git",
    "node_modules",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "target",
    ".idea",
    "_artifacts",
    "reports",
]
EXCLUDED_EXTENSIONS = [
    ".log",
    ".lock",
    ".svg",
    ".png",
    ".jpg",
    ".gif",
    ".ico",
    ".xml",
    ".json",
    ".mdc",
    ".mjs",
    ".xlsx",
]
EXCLUDED_FILES = [".env", ".gitignore", ".prettierrc"]


# --- 언어별 주석 패턴 정의 ---
LANGUAGE_COMMENT_PATTERNS = {
    # JavaScript, TypeScript, Java, C++, C#, etc.
    ".js": {"single_line": ["//"], "multi_line_start": "/*", "multi_line_end": "*/"},
    ".ts": {"single_line": ["//"], "multi_line_start": "/*", "multi_line_end": "*/"},
    ".java": {"single_line": ["//"], "multi_line_start": "/*", "multi_line_end": "*/"},
    ".cs": {"single_line": ["//"], "multi_line_start": "/*", "multi_line_end": "*/"},
    ".cpp": {"single_line": ["//"], "multi_line_start": "/*", "multi_line_end": "*/"},
    ".c": {"single_line": ["//"], "multi_line_start": "/*", "multi_line_end": "*/"},
    ".go": {"single_line": ["//"], "multi_line_start": "/*", "multi_line_end": "*/"},
    # Python
    ".py": {"single_line": ["#"], "multi_line_start": '"""', "multi_line_end": '"""'},
    # HTML, XML - 실제로는 지만, 단순화를 위해 우선 비워둠
    ".html": {"single_line": [], "multi_line_start": None, "multi_line_end": None},
    ".xml": {"single_line": [], "multi_line_start": None, "multi_line_end": None},
    # Shell Scripts
    ".sh": {"single_line": ["#"], "multi_line_start": None, "multi_line_end": None},
    # CSS
    ".css": {"single_line": [], "multi_line_start": "/*", "multi_line_end": "*/"},
}

# --- 청킹(Chunking) 설정 ---
CHUNK_SIZE_LINES = 100
CHUNK_OVERLAP_LINES = 10
