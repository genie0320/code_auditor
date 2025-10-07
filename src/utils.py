# src/utils.py
import json
from pathlib import Path
import config  # config 모듈 임포트


def path_serializer(obj):
    """Path 객체를 문자열로 변환하는 JSON 직렬화 헬퍼"""
    if isinstance(obj, Path):
        return str(obj.as_posix())  # Windows/Linux 호환성을 위해 as_posix() 사용
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_json(data: dict | list, file_path: Path):
    """주어진 데이터를 JSON 파일로 저장하는 유틸리티 함수."""
    print(f"[UTILS] Saving data to {file_path}...")

    # 디렉토리가 없으면 생성
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # JSON 파일 쓰기
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=path_serializer)
    print(f"[UTILS] Successfully saved.")


def load_json(file_path: Path) -> dict | list:
    """JSON 파일을 읽어 파이썬 객체(dict 또는 list)로 반환합니다."""
    print(f"[UTILS] Loading data from {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_lines(file_path: Path) -> dict:
    """
    파일을 읽어 언어별 주석 패턴에 따라 전체, 코드, 주석 라인 수를 분석.
    """

    extension = file_path.suffix
    patterns = config.LANGUAGE_COMMENT_PATTERNS.get(extension, {})

    single_line_patterns = patterns.get("single_line", [])
    multi_start = patterns.get("multi_line_start")
    multi_end = patterns.get("multi_line_end")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return {"total": 0, "code": 0, "comment": 0}

    total_lines = len(lines)
    comment_lines = 0
    code_lines = 0
    in_multiline_comment = False

    for line in lines:
        stripped_line = line.strip()

        if not stripped_line:
            continue  # 빈 줄은 무시

        is_comment_line = False

        # 멀티라인 주석 상태 처리
        if multi_start and multi_end:
            if in_multiline_comment:
                comment_lines += 1
                is_comment_line = True
                if multi_end in stripped_line:
                    in_multiline_comment = False
            elif stripped_line.startswith(multi_start):
                comment_lines += 1
                is_comment_line = True
                # 주석이 같은 줄에서 끝나지 않으면 멀티라인 상태 유지
                if not stripped_line.endswith(multi_end) or len(stripped_line) == len(
                    multi_start
                ):
                    in_multiline_comment = True

        # 한 줄 주석 처리
        if not is_comment_line and any(
            stripped_line.startswith(p) for p in single_line_patterns
        ):
            comment_lines += 1
            is_comment_line = True

        if not is_comment_line:
            code_lines += 1

    return {"total": total_lines, "code": code_lines, "comment": comment_lines}
