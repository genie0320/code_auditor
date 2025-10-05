# config.py

import os

# 무시할 폴더 및 파일 설정
DEFAULT_IGNORE_DIRS = {
    ".git",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".mypy_cache",
    ".venv",
}
DEFAULT_IGNORE_FILES = {
    ".DS_Store",
    "package-lock.json",
    "yarn.lock",
    "LICENSE",
    "README.md",
    "CONTRIBUTING.md",
}


def get_ignore_patterns() -> tuple[set, set]:
    """무시할 폴더와 파일 목록을 반환합니다."""
    return DEFAULT_IGNORE_DIRS, DEFAULT_IGNORE_FILES


def get_claude_api_key() -> str:
    """환경 변수에서 Claude API 키를 로드합니다. (현재는 PLACEHOLDER)"""
    # API 키는 4단계에서 실제 로직으로 변경됩니다.
    return "PLACEHOLDER_ANTHROPIC_API_KEY"
