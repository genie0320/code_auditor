1. 전체 디렉토리 구조 (The Blueprint)
   code-audit-service/
   ├── .env # API 키 등 민감 정보 저장
   ├── .gitignore
   ├── config.py # 모든 설정을 중앙 관리하는 설정 파일
   ├── main.py # 어플리케이션의 시작점 (Orchestrator 실행)
   ├── requirements.txt # 의존성 목록
   ├── reports/ # 감사 결과 JSON 파일이 저장될 디렉토리
   │ └── .gitkeep
   └── src/ # 핵심 소스 코드가 위치할 패키지
   ├── **init**.py
   ├── core/
   │ ├── **init**.py
   │ └── orchestrator.py # 전체 감사 워크플로우를 지휘
   ├── db/
   │ ├── **init**.py
   │ └── vector_db_handler.py # ChromaDB 상호작용 (Phase 2에서 활성화)
   ├── llm_clients/
   │ ├── **init**.py # LLM 클라이언트 선택을 위한 팩토리
   │ ├── base_client.py # 모든 LLM 클라이언트가 따라야 할 인터페이스(ABC)
   │ ├── claude_client.py
   │ ├── gemini_client.py
   │ └── openai_client.py
   └── services/
   ├── **init**.py
   ├── code_chunker.py # 코드 분할(청킹) 로직 담당
   ├── file_scanner.py # 감사 대상 파일 스캔 및 필터링
   └── report_generator.py # 감사 결과 리포트 생성 및 저장
