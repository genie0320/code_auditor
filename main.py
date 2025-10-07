# main.py
import time
from datetime import datetime

import config
from src.core.orchestrator import AuditOrchestrator
from src.services.file_scanner import FileScanner
from src.services.code_chunker import CodeChunker
from src.services.report_generator import ReportGenerator
from src.llm_clients import get_llm_client
from src.services.chunking_strategies import (
    LineChunkingStrategy,
    TreeSitterChunkingStrategy,
)


def main():
    """어플리케이션 초기화 및 실행"""
    # 각 디렉토리가 존재하는지 확인하고 없으면 생성
    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.ARTIFACTS_DIR / "chunks").mkdir(parents=True, exist_ok=True)
    job_id = datetime.now().strftime("%Y%m%d%H%M%S")  # 작업 ID 생성

    # (수정) 설정에 따라 적절한 청킹 전략 객체 생성
    if config.CHUNKING_STRATEGY == "line":
        chunking_strategy = LineChunkingStrategy(
            chunk_size=config.LINE_CHUNK_SIZE, overlap=config.LINE_CHUNK_OVERLAP
        )
    else:  # "treesitter"
        chunking_strategy = TreeSitterChunkingStrategy(
            chunk_size=config.TS_CHUNK_SIZE, overlap=config.TS_CHUNK_OVERLAP
        )

    # 1. 의존성 생성
    scanner = FileScanner(config.TARGET_ROOT_DIR, job_id)
    chunker = CodeChunker(strategy=chunking_strategy)
    reporter = ReportGenerator(config.REPORTS_DIR)
    llm_client = get_llm_client(
        client_name=config.SELECTED_LLM_CLIENT,
        api_key=config.LLM_API_KEYS.get(config.SELECTED_LLM_CLIENT),
        default_model=config.LLM_DEFAULT_MODEL,
    )

    # 2. 의존성 주입
    orchestrator = AuditOrchestrator(scanner, chunker, llm_client)

    # 3. 감사 실행
    orchestrator.run_audit()


if __name__ == "__main__":
    main()
