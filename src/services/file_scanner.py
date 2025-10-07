# src/services/file_scanner.py
import config
import uuid
from pathlib import Path
from src.models import FileContext
from src.utils import analyze_lines


class FileScanner:
    def __init__(self, root_dir: str, job_id: str):
        self.root_dir = Path(root_dir)
        self.job_id = job_id

        # 제외 목록 초기화
        self.excluded_folders = set(config.EXCLUDED_FOLDERS)
        self.excluded_extensions = set(config.EXCLUDED_EXTENSIONS)
        self.excluded_files = set(config.EXCLUDED_FILES)
        print(f"[FileScanner] Initialized. Job ID: {self.job_id}")

    def _is_excluded(self, file_path: Path) -> bool:
        """정교화된 제외 규칙을 적용하여 파일 필터링"""
        # 1. 파일 이름이 제외 목록에 있는지 확인
        if file_path.name in self.excluded_files:
            return True

        # 2. 파일 확장자가 제외 목록에 있는지 확인
        if file_path.suffix in self.excluded_extensions:
            return True

        # 3. 파일의 경로 중 일부가 제외 폴더 목록에 포함되는지 확인
        # (e.g., /path/to/node_modules/some/file.js)
        for part in file_path.parts:
            if part in self.excluded_folders:
                return True

        return False

    def scan(self) -> list[FileContext]:
        """파일을 스캔하고 각 파일에 대한 FileContext 객체 리스트를 생성."""
        print("[FileScanner] Scanning files and generating metadata...")

        file_contexts = []
        for file_path in self.root_dir.rglob("*"):
            if file_path.is_dir() or self._is_excluded(file_path):
                continue

            # (수정) 제외 여부를 먼저 판단하여 status 변수에 저장
            is_excluded = self._is_excluded(file_path)
            status = "excluded" if is_excluded else "included"

            # 제외된 파일도 FileContext는 생성하되, 라인 분석은 건너뛰어 효율화
            if is_excluded:
                line_counts = {"total": 0, "code": 0, "comment": 0}
            else:
                line_counts = analyze_lines(file_path)

            # FileContext 객체 생성
            context = FileContext(
                job_id=self.job_id,
                file_id=f"{self.job_id}_{uuid.uuid4().hex[:8]}",
                full_path=file_path,
                directory=file_path.parent,
                filename=file_path.name,
                extension=file_path.suffix,
                total_lines=line_counts["total"],
                code_lines=line_counts["code"],
                comment_lines=line_counts["comment"],
                status=status,
            )
            file_contexts.append(context)

        print(
            f"[FileScanner] Scan finished. Generated {len(file_contexts)} FileContext objects."
        )
        return file_contexts
