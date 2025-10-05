# utils.py

import os
import re
from typing import List, Dict, Set, Any, Tuple, Union
from rich.console import Console
from rich.table import Table


# (1) 파일 목록 스캔 및 클리닝
def scan_and_clean_codebase(
    root_dir: str, ignore_dirs: Set[str], ignore_files: Set[str]
) -> List[Dict[str, Any]]:
    """
    지정된 폴더/파일을 제외하고 코드베이스를 스캔하여 분석된 파일 데이터를 반환합니다.
    """
    file_data_list = []

    # 1. 로그 초기화
    log_content = f"--- 작업 클리닝 로그: {root_dir} ---\n"
    total_ignored_count = 0

    for dirpath, dirnames, filenames in os.walk(root_dir):

        # 1.1 디렉토리 클리닝: ignore_dirs 목록에 있는 디렉토리 제거
        dirnames_to_remove = set(dirnames) & ignore_dirs
        for dname in dirnames_to_remove:
            dirnames.remove(
                dname
            )  # os.walk가 다음 탐색에서 해당 디렉토리를 건너뛰게 함
            log_content += (
                f"IGNORING DIR: {os.path.join(dirpath, dname)} (자동 생성/불필요)\n"
            )
            total_ignored_count += 1

        for filename in filenames:
            file_path = os.path.join(dirpath, filename)

            # 1.2 파일 클리닝: ignore_files 목록에 있는 파일 제거
            if filename in ignore_files:
                log_content += f"IGNORING FILE: {file_path} (불필요한 빌드 파일)\n"
                total_ignored_count += 1
                continue

            # 2. 파일 내용 읽기
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    file_content = f.read()
            except UnicodeDecodeError:
                # 인코딩 오류 발생 시 로그 기록 후 건너뛰기
                log_content += f"IGNORING FILE: {file_path} (인코딩 오류 발생)\n"
                total_ignored_count += 1
                continue
            except Exception as e:
                log_content += f"IGNORING FILE: {file_path} (읽기 오류: {e})\n"
                total_ignored_count += 1
                continue

            # 3. 라인 분석 수행 및 결과 취합
            total_lines, code_lines, comment_lines = analyze_lines(file_content)

            file_data_list.append(
                {
                    "file_path": file_path,
                    "file_content": file_content,
                    "total_lines": total_lines,
                    "code_lines": code_lines,
                    "comment_lines": comment_lines,
                }
            )

    log_content += f"--- 클리닝 종료: 총 {total_ignored_count}개 항목 제외 ---\n"
    # save_log 함수는 main.py에서 호출할 것이므로 여기서 로그를 반환하지 않고,
    # log_content를 main에서 사용할 수 있도록 별도로 처리하는 것이 좋습니다.
    # 하지만 워크플로우를 간소화하기 위해 이 함수 내에서 로그를 출력하고, main.py에서 최종 로그를 처리하도록 하겠습니다.

    print(log_content)  # 터미널에 클리닝 로그 즉시 출력

    return file_data_list


# (2) 라인 분석 (이전에 구현했던 analyze_lines 헬퍼 함수)
def analyze_lines(file_content: str) -> tuple[int, int, int]:
    """
    주어진 텍스트를 분석하여 (total_lines, code_lines, comment_lines) 튜플을 반환합니다.
    """
    # TODO: 여러가지 언어에서 사용하는 주석처리 기호 목록을 받아서, 그에 해당 하는 것이 있는지를 구분해내는 로직 추가. 또는 언어별로 사용하는 각종 기호를 정리하여 언어설정시에 자동반영하는 로직 추가.

    # TODO: 제외할 팔이 목록 찾아서 넣기 [svg, png, xml 등]

    lines = file_content.splitlines()
    total_lines = len(lines)
    code_lines = 0
    comment_lines = 0
    in_multiline_comment = False
    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            continue
        if "/*" in stripped_line and "*/" not in stripped_line:
            in_multiline_comment = True
            comment_lines += 1
            continue
        if "*/" in stripped_line:
            comment_lines += 1
            in_multiline_comment = False
            continue
        if in_multiline_comment:
            comment_lines += 1
            continue
        is_comment = False
        if stripped_line.startswith("#") or stripped_line.startswith("//"):
            is_comment = True
        if is_comment:
            comment_lines += 1
        else:
            code_lines += 1
    return total_lines, code_lines, comment_lines


# (3) Rich 테이블 출력
def print_analysis_table_rich(file_data_list: List[Dict[str, Any]], console: Console):
    """
    분석된 파일 리스트를 Rich 테이블로 포맷팅하여 터미널에 출력하고 로그로 저장합니다.
    """
    # 총 라인 수 계산 로직
    total_lines = sum(item["total_lines"] for item in file_data_list)
    total_code_lines_sum = sum(item["code_lines"] for item in file_data_list)
    total_comment_lines_sum = sum(item["comment_lines"] for item in file_data_list)

    # Rich Table 데이터 생성
    table = Table(
        title=f"[bold cyan]코드베이스 라인 분석 (총 {total_lines} 라인)[/bold cyan]",
        show_header=True,
        header_style="bold magenta",
        title_justify="left",
    )

    headers = ["file_path", "file_name", "extention", "lines_of_code", "comment_lines"]
    column_styles = {
        "file_path": ("cyan", "left"),
        "file_name": ("green", "left"),
        "extention": ("yellow", "center"),
        "lines_of_code": ("bold blue", "right"),
        "comment_lines": ("dim", "right"),
    }

    # Rich Table 출력
    for header in headers:
        style, justify = column_styles.get(header, ("default", "left"))
        table.add_column(header.replace("_", " ").title(), style=style, justify=justify)

    for item in file_data_list:
        file_path = item["file_path"]
        file_name_with_ext = os.path.basename(file_path)
        _, extension = os.path.splitext(file_name_with_ext)

        table.add_row(
            file_path,
            file_name_with_ext,
            extension,
            str(item["code_lines"]),
            str(item["comment_lines"]),
        )

    # 총계 행 (TOTAL) 추가
    styled_total_values = [
        "[bold]TOTAL[/bold]",
        "",
        "",
        f"[bold]{total_code_lines_sum}[/bold]",
        f"[bold]{total_comment_lines_sum}[/bold]",
    ]
    table.add_row(*styled_total_values, style="bold white on #4B0082", end_section=True)

    console.print(table)

    # console.print(
    #     f"[bold green]✅ 분석 완료: 총 코드 라인 {total_code_lines}, 주석 라인 {total_comment_lines}[/bold green]"
    # )


# (4) 로그 저장 유틸리티
def save_log(filename: str, content: str, mode: str = "w"):
    """클리닝 로그 및 분석 테이블을 파일로 저장합니다."""
    try:
        with open(filename, mode, encoding="utf-8") as f:
            f.write(content + "\n")
        print(f"LOG: '{filename}'에 로그가 저장되었습니다.")
    except Exception as e:
        print(f"LOG ERROR: 파일 저장 중 오류 발생 ({filename}): {e}")


# (5) 파일 크기 형식 변환 헬퍼 (편의상 추가)
def format_bytes(size: int) -> str:
    """바이트 크기를 KB, MB 등으로 보기 좋게 변환합니다."""
    # ... 구현은 단순하므로 생략하고 PLACEHOLDER로 둡니다.
    return "X KB"  # PLACEHOLDER
