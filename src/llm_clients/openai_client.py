# src/llm_clients/openai_client.py
import json
from openai import OpenAI
from .base_client import LLMClient

import config


class OpenAIClient(LLMClient):
    def __init__(self, api_key: str, default_model: str, **kwargs):
        self.api_key = api_key
        self.model = default_model
        # (신규) OpenAI 클라이언트 인스턴스 초기화
        self.client = OpenAI(api_key=self.api_key)
        print(f"[OpenAIClient] Initialized. Model: {self.model}")

    def process_batch(self, code_chunks: list[str], system_prompt: str) -> list[dict]:
        results = []
        for chunk_content in code_chunks:
            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": chunk_content},
                ]

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    max_tokens=config.LLM_MAX_TOKENS,
                )

                # (수정) 응답 객체 전체를 dict로 변환하고, 추가 파싱 없이 그대로 저장
                full_response_dict = response.model_dump()
                results.append(full_response_dict)

            except Exception as e:
                results.append({"error": str(e)})

        return results

    # # (수정) placeholder 로직을 실제 API 호출 코드로 완전히 교체
    # def process_batch(self, code_chunks: list[str], system_prompt: str) -> list[dict]:
    #     print(
    #         f"[OpenAIClient] Processing batch of {len(code_chunks)} chunks via API..."
    #     )

    #     # 현재는 청크를 하나씩 처리하는 방식으로 구현 (추후 배치 API로 확장 가능)
    #     results = []
    #     for chunk_content in code_chunks:
    #         try:
    #             # LLM에 보낼 메시지 구성
    #             messages = [
    #                 {"role": "system", "content": system_prompt},
    #                 {"role": "user", "content": chunk_content},
    #             ]

    #             # OpenAI ChatCompletion API 호출
    #             response = self.client.chat.completions.create(
    #                 model=self.model,
    #                 messages=messages,
    #                 # JSON 형식으로 응답을 받도록 요청
    #                 response_format={"type": "json_object"},
    #             )

    #             # 응답 결과 파싱
    #             # response.choices[0].message.content가 JSON 문자열이므로 파싱
    #             audit_result = json.loads(response.choices[0].message.content)
    #             results.append(audit_result)

    #         except Exception as e:
    #             print(f"[OpenAIClient] Error processing chunk: {e}")
    #             # 에러 발생 시 빈 결과 추가
    #             results.append({"error": str(e)})

    #     return results
