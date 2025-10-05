# utils.py

from typing import List, Dict, Set, Any
from rich.console import Console
from rich.table import Table


# (1) 파일 목록 스캔 및 클리닝
def scan_and_clean_codebase(
    root_dir: str, ignore_dirs: Set[str], ignore_files: Set[str]
) -> List[Dict[str, Any]]:
    """
    지정된 폴더/파일을 제외하고 코드베이스를 스캔하여 파일 데이터를 반환합니다.
    제외된 항목은 로그에 저장하는 로직이 추가되어야 합니다.
    """
    print("LOG: 제외 목록을 로그에 기록하는 PLACEHOLDER")
    return [
        {
            "file_path": "path/to/file1.py",
            "file_content": "...",
            "total_lines": 50,
            "code_lines": 40,
            "comment_lines": 10,
        },
        # ... 실제 스캔 및 클리닝 로직 구현 필요
    ]


# (2) 라인 분석 (이전에 구현했던 analyze_lines 헬퍼 함수)
def analyze_lines(file_content: str) -> tuple[int, int, int]:
    """파일 내용을 분석하여 (total, code, comment) 라인 수를 반환합니다."""
    # 실제 분석 로직 구현 필요
    return 10, 8, 2  # PLACEHOLDER


# (3) Rich 테이블 출력
def print_analysis_table_rich(file_data_list: List[Dict[str, Any]], console: Console):
    """
    분석된 파일 리스트를 Rich 테이블로 포맷팅하여 터미널에 출력하고 로그로 저장합니다.
    """
    # 총 라인 수 계산 로직
    total_code_lines = sum(item["code_lines"] for item in file_data_list)
    total_comment_lines = sum(item["comment_lines"] for item in file_data_list)

    # Rich Table 생성 및 출력 로직 구현 필요
    console.print(
        f"[bold green]✅ 분석 완료: 총 코드 라인 {total_code_lines}, 주석 라인 {total_comment_lines}[/bold green]"
    )
    # 로그 저장 로직 구현 필요


# (4) 로그 저장 유틸리티
def save_log(filename: str, content: str):
    """클리닝 로그 및 분석 테이블을 파일로 저장합니다."""
    # 파일 저장 로직 구현 필요
    print(f"LOG: {filename} 저장 PLACEHOLDER")
