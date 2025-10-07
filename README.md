# 대화형 코드베이스 분석

## 전체 디렉토리 구조 (The Blueprint)

```
   code-audit-service/
   ├── .env
   ├── _artifacts/
   ├── chroma_db/
   ├── reports/
   ├── src/ # 핵심 소스 코드가 위치할 패키지
   │  ├── **init**.py
   │  ├── ...
   │  └── tools/            # (신규) AI 에이전트가 사용할 도구들을 모아둘 곳
   │    ├── __init__.py
   │    ├── file_system.py    # 파일 검색, 읽기 등 파일 시스템 관련 도구
   │    ├── git.py            # Git 히스토리 검색 등 Git 관련 도구
   │    └── rag.py            # 기존 RAG(유사도 검색) 기능을 활용하는 도구
   ├── ask.py                  # (신규) 대화형 분석 기능을 위한 진입점
   ├── main.py
   └── ...
```
## PLAN & TODO
- [ ] Project_spec 작성 알고리즘 다듬기.

## '도구 제작' 작업 계획안
각 연장은 하나의 명확한 기능을 가짐.

### 2. 제작할 도구 목록 및 명세(Specification)
각 파일에 어떤 '연장'을 만들지, 그 '연장'의 사용법(함수 시그니처)과 역할을 명확히 정의.

#### A. 파일 시스템 도구 (src/tools/file_system.py)
**search_files(pattern: str, exclude_pattern: str | None = None) -> list[str]**

역할: 
- [x] 프로젝트 전체에서 특정 파일 이름 패턴(예: **/migrations/*.ts)과 일치하는 파일 목록을 찾는다. 
- [ ] 제외 패턴을 지정하면 검색 결과에서 뺄 수도 있다.

**read_file_content(file_path: str) -> str**

역할: 
- [ ] 주어진 파일 경로의 전체 텍스트 내용을 읽어서 반환.

#### B. Git 도구 (src/tools/git.py)
**search_git_log(days: int, query: str | None = None) -> str**

역할: 
- [ ] 최근 days일 동안의 git log를 조회. 
- [ ] query가 주어지면, 해당 키워드가 포함된 커밋 메시지만 필터링하여 반환.
   - "테이블 구조 변경" 같은 질문에 답하기 위한 핵심 도구

#### C. RAG 도구 (src/tools/rag.py)
**find_similar_code(code_snippet: str, k: int = 5) -> list[dict]**

역할: 
Phase 1에서 이미 구축한 ChromaDB를 사용, 주어진 코드 조각(code_snippet)과 가장 유사한 코드 청크 k개를 찾아 반환. (기존의 RAG 파이프라인을 재사용)

### 3. 전체 작업 흐름
- ask.py (신규 진입점): main.py와 별도로, 사용자의 자연어 질문을 입력받는 진입로 마련.(추후통합)
   - 도구 초기화: ask.py는 시작될 때 src/tools/에 있는 모든 함수들을 LangChain의 Tool 객체로 감싸서 목록을 생성.

   - 에이전트 생성: 이 '도구 목록'과 '두뇌' LLM을 결합하여 LangChain 에이전트를 생성.

   - 질문 전달: ask.py가 사용자의 질문을 에이전트에게 전달.

```
에이전트 쪽에서 '생각 / 행동':
에이전트는 질문을 해석하고, 자신이 가진 도구 목록 중에서 가장 적절한 도구를 선택.
(예: "테이블 변경" -> search_git_log 또는 search_files 선택)
```

- 에이전트가 선택한 도구를 필요한 인자와 함께 실행하고, 
   - 그 결과를 전달.
   - 결과가 충분하지 않으면, 다른 도구를 추가로 사용. 
   (예: search_files로 찾은 파일 내용을 read_file_content로 읽기)

- 모든 정보가 모이면, 최종적으로 자연어 답변을 생성하여 사용자에게 노출.

