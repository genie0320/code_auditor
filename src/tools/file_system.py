# src/tools/file_system.py

"""파일 시스템을 조회하고 읽는 도구들"""

import glob
from pathlib import Path


# (수정) 'directory' 인자 추가
def search_files(directory: str, pattern: str) -> list[str]:
    """프로젝트 내에서 glob 패턴을 사용하여 파일 경로를 검색합니다."""
    print(
        f"--- Tool Called: search_files(directory='{directory}', pattern='{pattern}') ---"
    )

    # (수정) 검색 시작 위치를 Path(".") 대신 Path(directory)로 변경
    base_path = Path(directory)
    # recursive=True를 통해 하위 폴더까지 모두 검색
    return [str(p.relative_to(base_path)) for p in base_path.glob(pattern)]


# (수정) 'base_dir' 인자 추가
def read_file_content(base_dir: str, file_path: str) -> str:
    """주어진 파일 경로의 전체 텍스트 내용을 읽어서 반환합니다."""
    print(f"--- Tool Called: read_file_content(file_path='{file_path}') ---")
    try:
        # (수정) 기준 디렉토리와 상대 경로를 조합하여 전체 경로 생성
        full_path = Path(base_dir) / file_path
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file {file_path}: {e}"
