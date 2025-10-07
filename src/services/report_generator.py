# src/services/report_generator.py
import json
from pathlib import Path
import src.utils as utils


class ReportGenerator:
    def __init__(self, report_dir: Path):
        self.report_dir = report_dir
        print(
            f"[ReportGenerator] Initialized. Reports will be saved to: {self.report_dir}"
        )

    # TODO: 개별 파일 리포트 저장 기능은 당장은 필요 없어서 주석 처리
    # def save_report(self, file_context: FileContext):
    #     """FileContext에 담긴 감사 결과를 실제 JSON 파일로 저장."""
    #     report_file_name = f"{file_context.filename}_{file_context.file_id}.json"
    #     report_path = self.report_dir / report_file_name
    #     print(
    #         f"[ReportGenerator] Saving report for {file_context.filename} to {report_path}..."
    #     )

    #     print(f"--- Report Content for {file_context.filename} ---")
    #     # FileContext 객체를 dict로 변환하여 저장
    #     report_data = dataclasses.asdict(file_context)

    #     try:
    #         with open(report_path, "w", encoding="utf-8") as f:
    #             json.dump(report_data, f, indent=2, ensure_ascii=False)
    #         print(f"[ReportGenerator] Successfully saved report.")
    #     except Exception as e:
    #         print(
    #             f"[ReportGenerator] Error saving report for {file_context.filename}: {e}"
    #         )

    def generate_summary_report(
        self, job_id: str, project_db_path: Path, results_dir: Path
    ):
        """여러 중간 산출물을 종합하여 단일 최종 리포트를 생성."""
        report_path = self.report_dir / f"report_{job_id}.json"
        print(f"[ReportGenerator] Generating summary report to {report_path}...")

        # Placeholder: 실제로는 각 파일을 읽고 내용을 종합하는 로직 구현
        summary_data = {
            "job_id": job_id,
            "project_db_file": str(project_db_path),
            "total_findings": len(list(results_dir.glob("*.json"))),
            "summary": "This is a placeholder summary. TODO: Aggregate all findings.",
        }

        # 안전한 저장을 위해 utils.save_json 사용
        utils.save_json(summary_data, report_path)
