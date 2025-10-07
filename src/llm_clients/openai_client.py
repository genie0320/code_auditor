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

                # 모델에 따라서 json 반환을 지원하지 않는 것도 있다고 함(빨간줄)
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
