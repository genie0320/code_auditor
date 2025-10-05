# Plans

## Step 01 : File setting

1. '오프라인에서 안전하게 데이터를 분석하는 작업'을 먼저 완성한 후
2. '외부와 통신하는 위험 요소'를 마지막에 추가

|   순서    | 파일                    | 작업의 목적                                                                                | 난이도 |
| :-------: | :---------------------- | :----------------------------------------------------------------------------------------- | :----: |
| **1단계** | `utils.py`              | 핵심 로직 구현: 파일 스캔, 클리닝, 라인 분석, Rich 테이블 출력 (**데이터 처리 능력 확보**) |  중간  |
| **2단계** | `main.py`               | 워크플로우 테스트: 1단계 로직을 조합하여 터미널에 분석 결과가 정확하게 출력되는지 확인     |  낮음  |
| **3단계** | `claude_api_client.py`  | API 통신 인터페이스 구현: 배치 입력 파일 생성 로직 완성                                    |  중간  |
| **4단계** | `main.py` & `config.py` | 최종 통합 및 실행: API 키 설정, 배치 작업 제출 및 결과 처리 (**최종 목표 달성**)           |  중간  |

## Step 02 : Functions Setting with place holder

| 파일/클래스 | 함수/메서드                 | 인자 (Arguments)                                               | 반환 형태 (Return Type) | 최종 역할                                           |
| :---------- | :-------------------------- | :------------------------------------------------------------- | :---------------------- | :-------------------------------------------------- |
| `config.py` | `get_ignore_patterns`       | `None`                                                         | `tuple[set, set]`       | 기본 제외 폴더/파일 목록 반환                       |
| `config.py` | `get_claude_api_key`        | `None`                                                         | `str`                   | Claude API 키 반환 (Placeholder)                    |
| `utils.py`  | `analyze_lines`             | `file_content: str`                                            | `Tuple[int, int, int]`  | 코드/주석 라인 수 분석                              |
| `utils.py`  | `scan_and_clean_codebase`   | `root_dir: str, ignore_dirs: Set[str], ignore_files: Set[str]` | `List[Dict[str, Any]]`  | 파일 스캔, 클리닝, 라인 분석 후 리스트 반환         |
| `utils.py`  | `print_analysis_table_rich` | `file_data_list: List[Dict[str, Any]], console: Console`       | `None`                  | 분석 결과를 Rich 테이블로 출력                      |
| `utils.py`  | `save_log`                  | `filename: str, content: str, mode: str = 'w'`                 | `None`                  | 텍스트 콘텐츠를 파일에 저장                         |
| `utils.py`  | `format_bytes`              | `size: int`                                                    | `str`                   | 바이트 크기를 사람이 읽기 쉽게 포맷팅 (PLACEHOLDER) |
| `main.py`   | `run_batch_workflow`        | `source_folder: str`                                           | `None`                  | 전체 워크플로우 실행 및 오케스트레이션              |

## Step 03 : Set claude client

|       파일/클래스        |         함수/메서드          | 인자 (Arguments)                             | 반환 형태 (Return Type) | 최종 역할                                                       |
| :----------------------: | :--------------------------: | :------------------------------------------- | :---------------------: | :-------------------------------------------------------------- |
|       `config.py`        |     `get_claude_api_key`     | `None`                                       |          `str`          | Claude API 키 반환 (Placeholder)                                |
| **claude_api_client.py** | `ClaudeBatchClient` (Class)  | -                                            |         (Class)         | 배치 API 통신을 위한 클래스                                     |
|                          |          `__init__`          | `api_key: str`                               |         `None`          | 클라이언트 초기화                                               |
|                          |  `create_batch_input_file`   | `target_files: List[Dict], output_path: str` |          `str`          | **[구체화]** 분석 대상 파일로 JSONL 입력 파일 생성 및 경로 반환 |
|                          |      `submit_batch_job`      | `input_file_path: str, model: str`           |          `str`          | 배치 작업 제출 (Placeholder)                                    |
|                          | `check_and_download_results` | `job_id: str`                                |         `bool`          | 결과 확인 및 다운로드 (Placeholder)                             |
