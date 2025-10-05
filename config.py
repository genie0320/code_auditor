# config.py

# 무시할 폴더 및 파일 설정
DEFAULT_IGNORE_DIRS = {".git", "__pycache__", "node_modules", "dist", "build"}
DEFAULT_IGNORE_FILES = {
    ".DS_Store",
    "package-lock.json",
    "yarn.lock",
}
# TODO: 제외할 확장자 추가.


def get_ignore_patterns() -> tuple[set, set]:
    """무시할 폴더와 파일 목록을 반환합니다."""
    # 환경 변수나 설정 파일을 읽어와 확장 가능
    return DEFAULT_IGNORE_DIRS, DEFAULT_IGNORE_FILES


def get_claude_api_key() -> str:
    """환경 변수에서 Claude API 키를 로드합니다."""
    # os.environ.get('ANTHROPIC_API_KEY') 로 대체
    return "PLACEHOLDER_LOAD_API_KEY"
