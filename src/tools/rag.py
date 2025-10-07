# src/tools/rag.py

"""기존 RAG 파이프라인을 활용하여 유사 코드를 검색하는 도구"""

# 참고: rag.py는 ChromaDB와 Embedding 초기화가 필요하여 ask.py에서 직접 설정하는 것이 더 효율적이므로, 별도 파일은 생략.
# def find_similar_code(code_snippet: str, k: int = 5) -> list[dict]:
#     """주어진 코드 조각과 가장 유사한 코드 청크 k개를 벡터 DB에서 찾아 반환합니다."""
#     print(f"--- Tool Called: find_similar_code(k={k}) ---")
#     # Placeholder: 실제로는 ChromaDB에 연결하고 유사도 검색을 수행
#     return [{"similarity_score": 0.95, "chunk": {"chunk_content": "similar code..."}}]
