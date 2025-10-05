# claude_api_client.py

from typing import List, Dict


class ClaudeBatchClient:
    """Claude Batch API와의 통신을 관리하는 클라이언트 클래스."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        print(f"CLIENT: Claude Batch Client가 API 키와 함께 초기화되었습니다.")

    # def create_batch_input_file(
    #     self, target_files: List[Dict[str, any]], output_path: str
    # ) -> str:
    #     """
    #     Claude Batch API에 전달할 JSONL 형식의 입력 파일을 생성합니다.
    #     각 파일의 file_content를 기반으로 요청 메시지를 만듭니다.
    #     """
    #     # JSONL 파일 생성 로직 구현 필요
    #     print(
    #         f"API: Batch 입력 파일 '{output_path}' 생성을 요청받았습니다. PLACEHOLDER"
    #     )
    #     return output_path

    def create_batch_input_file(
        self, target_files: List[Dict[str, Any]], output_path: str
    ) -> str:
        """
        [업데이트] Claude Batch API에 전달할 JSONL 형식의 입력 파일을 생성하고 저장합니다.
        각 파일의 file_content를 기반으로 요청 메시지를 만듭니다.
        """
        import json  # 함수 내부에서 json 모듈을 임포트합니다.

        # 1. 시스템 프롬프트 정의: 모든 파일에 동일하게 적용할 지침
        # 이 프롬프트는 분석 요청에 따라 변경될 수 있습니다.
        system_prompt = (
            "You are an expert security and performance code reviewer. "
            "Analyze the following code snippet for potential security vulnerabilities, "
            "performance bottlenecks, and suggest best practice refactoring. "
            "Respond only with Markdown formatted analysis."
        )

        # 2. JSONL 파일 생성
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                for item in target_files:
                    # 2.1 Claude에게 보낼 최종 메시지 구성
                    # 파일 경로와 내용이 포함된 사용자 메시지입니다.
                    user_message = (
                        f"Please review the following file: {item['file_path']}\n"
                        f"Content:\n```\n{item['file_content']}\n```"
                    )

                    # 2.2 Claude Batch API 요청 형식 (Messages API 구조를 따름)
                    batch_request_data = {
                        "messages": [{"role": "user", "content": user_message}],
                        "system": system_prompt,  # 시스템 프롬프트를 별도로 지정
                        "model": "claude-3-opus",  # 배치 작업에 사용할 모델
                        "metadata": {
                            "file_path": item["file_path"],
                            "code_lines": item[
                                "code_lines"
                            ],  # 추적을 위한 메타데이터 포함
                        },
                    }

                    # 2.3 JSONL 형식으로 파일에 한 줄씩 작성
                    f.write(json.dumps(batch_request_data) + "\n")

            print(
                f"API: Batch 입력 파일 '{output_path}' 생성이 완료되었습니다. 총 {len(target_files)}개 작업."
            )
            return output_path

        except Exception as e:
            print(f"API ERROR: JSONL 파일 생성 중 오류 발생: {e}")
            raise

    def submit_batch_job(self, input_file_path: str, model: str) -> str:
        """
        생성된 입력 파일로 배치 작업을 제출하고 Job ID를 반환합니다.
        """
        # Anthropic API 호출 및 job_id 반환 로직 필요
        job_id = "job-xyz-123"
        print(f"API: 배치 작업이 제출되었습니다. Job ID: {job_id} PLACEHOLDER")
        return job_id

    def check_and_download_results(self, job_id: str):
        """
        배치 작업 상태를 확인하고 완료되면 결과를 다운로드합니다.
        """
        # 상태 폴링 및 결과 다운로드 로직 필요
        print(
            f"API: Job {job_id}의 결과 확인 및 다운로드 로직이 실행됩니다. PLACEHOLDER"
        )
        return True
