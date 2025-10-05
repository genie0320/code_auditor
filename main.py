# main.py

import os
from rich.console import Console

# 모듈 임포트
from config import get_ignore_patterns, get_claude_api_key
from utils import scan_and_clean_codebase, print_analysis_table_rich, save_log
from claude_api_client import ClaudeBatchClient


def run_batch_workflow(source_folder: str):
    """
    배치 작업을 위한 전체 워크플로우를 실행합니다.
    """
    console = Console()

    # 템프 폴더 생성 (로그 및 배치 입력 파일 저장을 위해)
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)

    console.print(
        f"[bold yellow]=== Claude Batch 워크플로우 시작: {source_folder} ===[/bold yellow]"
    )

    # 1. 아키텍처 설정 로드
    api_key = get_claude_api_key()
    ignore_dirs, ignore_files = get_ignore_patterns()

    # claude_api_client 객체 초기화
    client = ClaudeBatchClient(api_key=api_key)

    # ===================================================
    # 단계 1: 작업 클리닝 및 분석
    # ===================================================

    console.print("\n[bold]1. 코드베이스 스캔 및 클리닝...[/bold]")

    # 1.1 스캔 및 클리닝 (로그 기록 포함)
    target_files_data = scan_and_clean_codebase(
        source_folder, ignore_dirs, ignore_files
    )

    # 1.2 라인 분석 및 터미널/로그 노출
    if not target_files_data:
        console.print(
            "[bold red]❌ 작업 대상 파일이 없습니다. 워크플로우를 종료합니다.[/bold red]"
        )
        return

    # 1.2 라인 분석 결과 Rich 테이블로 터미널에 출력
    console.print("\n[bold]2. 분석 결과 및 작업 대상 파일 목록[/bold]")
    print_analysis_table_rich(target_files_data, console)

    # 1.3 최종 로그 파일 저장
    total_code_lines = sum(item["code_lines"] for item in target_files_data)

    # TODO: 분석 테이블을 문자열로 변환하여 저장하는 util 추가 (현재 약식으로 저장)
    log_summary = f"총 {len(target_files_data)}개 파일 스캔 완료. 최종 코드 라인 수: {total_code_lines}"
    save_log(os.path.join(temp_dir, "workflow_summary.log"), log_summary)

    # ===================================================
    # 단계 2: Claude Batch API 호출
    # ===================================================

    console.print("\n[bold]3. Claude Batch API 작업 준비 및 제출 Placeholder...[/bold]")

    # 2.1 배치 입력 파일 생성
    input_jsonl_path = os.path.join("temp", "batch_input.jsonl")
    client.create_batch_input_file(target_files_data, input_jsonl_path)

    # 2.2 배치 작업 제출
    # TODO: Sonnet의 정확한 모델명. 모델을 변경할 수 있도록 config에 옵션 추가.
    job_id = client.submit_batch_job(
        input_jsonl_path, model="claude-3-opus"
    )  # 모델 설정

    # 2.3 결과 확인 및 다운로드 (비동기)
    console.print(
        "\n[bold green]▶️ 배치 작업이 제출되었습니다. 결과 다운로드를 시작합니다...[/bold green]"
    )
    client.check_and_download_results(job_id)

    console.print(
        f"\n[bold yellow]=== 워크플로우 완료: 임시 파일은 {temp_dir} 폴더에 저장되었습니다. ===[/bold yellow]"
    )


if __name__ == "__main__":
    # 사용자가 설정할 '보낼 코드의 소스파일폴더'
    SOURCE_CODE_ROOT = "./_source_code"
    run_batch_workflow(SOURCE_CODE_ROOT)
