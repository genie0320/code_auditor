# src/core/orchestrator.py

# --- 1. 표준 라이브러리 임포트 ---
import dataclasses
import datetime
import glob
from pathlib import Path

# --- 2. 서드파티 라이브러리 임포트 ---
import config
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma  # (수정) LangChain 최신 버전에 맞춰 경로 변경

# --- 3. 로컬 애플리케이션 임포트 ---
from src import utils
from src.llm_clients.base_client import LLMClient
from src.services.file_scanner import FileScanner
from src.services.code_chunker import CodeChunker
from src.services.report_generator import ReportGenerator
from src.services.batch_builder import BatchBuilder


class AuditOrchestrator:
    def __init__(
        self, scanner: FileScanner, chunker: CodeChunker, llm_client: LLMClient
    ):
        self.scanner = scanner
        self.chunker = chunker
        self.llm_client = llm_client
        # self.report_generator = report_generator
        print("[Orchestrator] Initialized with all components.")

    def _get_system_prompt(self) -> str:
        return "You are a code auditor. Find vulnerabilities. Respond in JSON."

    def _format_batch_prompt(self, batch: dict, file_id_map: dict) -> dict:
        """배치 정보를 받아 LLM에 보낼 최종 프롬프트 패키지를 생성합니다."""

        system_prompt = """
            You are an expert code auditor specializing in finding code duplication and suggesting refactorings.
            Below are several code chunks that have been identified as highly similar.
            Analyze them together, identify the common pattern, and suggest a single, improved version that could replace all of them.
            Respond ONLY in JSON format with the following structure: {"analysis": "...", "recommendation": "..."}.
            """

        user_content_parts = []
        chunk_metadata = []

        for i, chunk_info in enumerate(batch["chunks"]):
            # chunk_info에서 실제 청크 데이터가 담긴 dict를 꺼냄
            chunk_dict = chunk_info["chunk"]
            parent_file_id = chunk_dict["parent_file_id"]
            file_path = file_id_map.get(parent_file_id, "Unknown File")

            file_info = f"File: {file_path}, Lines: {chunk_dict['start_line']}-{chunk_dict['end_line']}"

            user_content_parts.append(
                f"--- Similar Code Chunk {i+1} ({file_info}) ---\n{chunk_dict['chunk_content']}"
            )
            chunk_metadata.append(
                {"chunk_id": chunk_dict["chunk_id"], "parent_file_id": parent_file_id}
            )

        user_prompt = "\n\n".join(user_content_parts)

        return {
            "batch_id": batch["batch_id"],
            "metadata": {
                "chunk_count": len(batch["chunks"]),
                "created_at": datetime.datetime.now().isoformat(),
                "included_chunks": chunk_metadata,
            },
            "prompt": {"system": system_prompt, "user": user_prompt},
        }

    def run_audit(self):
        print("\n" + "=" * 20 + " Audit Start " + "=" * 20)
        job_id = self.scanner.job_id

        # ------------------------------------------------------------
        # --- Step 1: 파일 메타데이터 생성 ---
        all_file_contexts = self.scanner.scan()
        project_db_path = config.ARTIFACTS_DIR / f"project_db_{job_id}.json"
        utils.save_json(
            [dataclasses.asdict(ctx) for ctx in all_file_contexts], project_db_path
        )
        print(f"Step 1 Done: Project DB saved to {project_db_path}")

        # ------------------------------------------------------------
        # --- Step 2: 코드 청킹 및 토큰 계산 ---
        files_to_chunk = [ctx for ctx in all_file_contexts if ctx.status == "included"]

        # 중복 ID를 걸러내기 위해 dictionary를 사용
        unique_chunks_map = {}
        for context in files_to_chunk:
            file_chunks = self.chunker.chunk_file(context)
            for chunk in file_chunks:
                # chunk_id를 key로 사용하여 덮어쓰면 자연스럽게 중복이 제거됨
                unique_chunks_map[chunk.chunk_id] = chunk

        # dictionary의 값들만 리스트로 변환하여 최종 all_chunks 생성
        all_chunks = list(unique_chunks_map.values())

        chunks_db_path = config.ARTIFACTS_DIR / f"chunks_{job_id}.json"
        utils.save_json([dataclasses.asdict(c) for c in all_chunks], chunks_db_path)
        print(f"Step 2 Done: Chunks DB saved to {chunks_db_path}")

        # 청크가 하나도 없으면 이후 단계를 진행하지 않고 종료
        if not all_chunks:
            print(
                "No chunks were generated to audit. Skipping embedding and further steps."
            )
            # 최종 리포트 생성 단계로 바로 넘어갈 수도 있지만, 여기서는 정상 종료 처리
            print("\n" + "=" * 20 + " Audit End (No files to process) " + "=" * 22)
            return

        # ------------------------------------------------------------
        # --- Step 3: 임베딩 및 벡터 DB 저장 (중복 제거 로직 추가) ---
        print("Step 3 Start: Embedding new chunks and saving to Vector DB...")
        # (개선) 1. 어떤 임베딩을 사용할지 먼저 결정
        if config.EMBEDDING_PROVIDER == "local":
            print(f"Using local embedding model: {config.EMBEDDING_MODEL_NAME}")
            embeddings = HuggingFaceEmbeddings(
                model_name=config.EMBEDDING_MODEL_NAME,
                model_kwargs={"device": "cuda"},  # cpu
                encode_kwargs={"normalize_embeddings": True},
            )
        else:  # "openai"
            api_key = config.LLM_API_KEYS.get("openai")
            if not api_key or "placeholder" in api_key:
                raise ValueError(
                    "OpenAI API key is not configured correctly for embedding."
                )
            embeddings = OpenAIEmbeddings(
                model=config.OPENAI_EMBEDDING_MODEL, api_key=api_key
            )

        # (개선) 2. Chroma 객체를 단 한 번만 생성
        vector_store = Chroma(
            persist_directory=config.CHROMA_DB_PATH, embedding_function=embeddings
        )

        # DB에 이미 저장된 모든 청크 ID를 가져오기
        existing_ids = set(vector_store.get()["ids"])
        print(f"Found {len(existing_ids)} existing chunks in DB.")

        # (신규) 새로 생성된 청크 중, DB에 아직 없는 청크만 필터링
        chunks_to_embed = [
            chunk for chunk in all_chunks if chunk.chunk_id not in existing_ids
        ]
        print(f"Found {len(chunks_to_embed)} new or modified chunks to embed.")

        if chunks_to_embed:
            documents = [
                Document(
                    page_content=chunk.chunk_content,
                    metadata={
                        "chunk_id": chunk.chunk_id,
                        "parent_file_id": chunk.parent_file_id,
                        "chunk_method": chunk.chunk_method,
                    },
                    # (신규) 문서 ID를 우리의 결정적 chunk_id로 직접 지정
                    id=chunk.chunk_id,
                )
                for chunk in chunks_to_embed
            ]

            # (수정) from_documents 대신 add_documents를 사용하여 기존 DB에 데이터 추가
            vector_store = Chroma(
                persist_directory=config.CHROMA_DB_PATH, embedding_function=embeddings
            )
            vector_store.add_documents(
                documents, ids=[chunk.chunk_id for chunk in chunks_to_embed]
            )
            print(f"Added {len(documents)} new chunks to Vector DB.")
        else:
            print("No new chunks to add. Skipping embedding.")

        print(f"Step 3 Done: Vector DB is up to date at {config.CHROMA_DB_PATH}")

        # ------------------------------------------------------------
        # --- Step 4: 유사 청크 그룹핑 및 배치 생성 (델타 업데이트 적용) ---
        print("Step 4 Start: Building audit batches...")

        previous_batches = None
        # job_id 패턴으로 가장 최신 배치 파일 찾기
        batch_files = sorted(
            glob.glob(str(config.ARTIFACTS_DIR / "batches_*.json")), reverse=True
        )
        if batch_files:
            latest_batch_file = Path(batch_files[0])
            print(f"Found previous batch file: {latest_batch_file.name}")
            try:
                # utils.py에 load_json 함수가 구현되어 있어야 합니다.
                previous_batches = utils.load_json(latest_batch_file)
            except Exception as e:
                print(
                    f"Warning: Could not load previous batch file. Full build will be performed. Error: {e}"
                )

        batch_builder = BatchBuilder(vector_store, all_chunks)
        new_chunk_ids = {chunk.chunk_id for chunk in chunks_to_embed}

        batches = batch_builder.build_batches(
            similarity_threshold=config.SIMILARITY_THRESHOLD,
            previous_batches=previous_batches,
            new_chunk_ids=new_chunk_ids,
        )

        batches_db_path = config.ARTIFACTS_DIR / f"batches_{job_id}.json"
        utils.save_json(batches, batches_db_path)
        print(
            f"Step 4 Done: Found {len(batches)} batches. DB saved to {batches_db_path}"
        )

        # ------------------------------------------------------------
        # --- Step 5: LLM 감사 및 결과 저장 ---
        # print("Step 5 Start: Preparing prompts and auditing batches via LLM...")
        print("Step 5 Start: Auditing 'ready' batches and updating status...")

        # (신규) parent_file_id를 실제 파일 경로로 변환하기 위한 맵 생성
        file_id_to_path_map = {
            ctx.file_id: str(ctx.full_path) for ctx in all_file_contexts
        }
        # results 폴더가 없으면 생성
        results_dir = config.ARTIFACTS_DIR / "results"
        prompts_dir = config.ARTIFACTS_DIR / "prompts"
        results_dir.mkdir(exist_ok=True)
        prompts_dir.mkdir(exist_ok=True)

        # (수정) 'ready' 상태인 배치만 필터링하여 감사 대상으로 삼음
        ready_batches = [b for b in batches if b.get("status") == "ready"]
        print(f"Found {len(ready_batches)} batches in 'ready' state to process.")

        # (수정) 배치가 있을 경우에만 진행
        if ready_batches:
            for batch in ready_batches:
                try:
                    # (신규) 가장 많은 청크를 가진 배치를 테스트 대상으로 선택
                    batch.sort(key=lambda b: len(b["chunks"]), reverse=True)
                    target_batch = batch[0]
                    print(
                        f"Selected largest batch '{target_batch['batch_id']}' with {len(target_batch['chunks'])} chunks for the test."
                    )

                    # 5a: 프롬프트 패키지 구성 및 파일 저장 (미리보기)
                    prompt_package = self._format_batch_prompt(
                        target_batch, file_id_to_path_map
                    )
                    prompt_preview_path = (
                        prompts_dir / f"{target_batch['batch_id']}.json"
                    )
                    utils.save_json(prompt_package, prompt_preview_path)

                    # 5b: (수정) 실제 LLM API 호출 활성화!
                    print(f"Auditing batch {target_batch['batch_id']}...")
                    audit_results = self.llm_client.process_batch(
                        [
                            prompt_package["prompt"]["user"]
                        ],  # 사용자 프롬프트는 하나로 합쳐져 있음
                        prompt_package["prompt"]["system"],
                    )
                    result_path = results_dir / f"{target_batch['batch_id']}.json"
                    utils.save_json(audit_results, result_path)

                    # (수정) 성공 시 상태를 'done'으로 변경
                    batch["status"] = "done"
                    print(f"  -> Batch {batch['batch_id']} status updated to 'done'.")
                    print(
                        f"Step 5 Done: Audit result for one batch saved to {result_path}"
                    )
                except Exception as e:
                    print(f"  -> ERROR auditing batch {batch['batch_id']}: {e}")
                    # (수정) 실패 시 상태를 'fail'로 변경하고 에러 메시지 기록
                    batch["status"] = "fail"
                    batch["error_message"] = str(e)

            # (신규) 모든 작업이 끝난 후, 업데이트된 상태가 포함된 전체 배치 목록을 다시 저장
            utils.save_json(batches, batches_db_path)
            print(f"Step 5 Done: Batch statuses updated and saved to {batches_db_path}")

        else:
            print("Step 5 Skipped: No batches were created.")

        # for batch in batches:
        #     # 5a: LLM에 보낼 프롬프트 패키지를 구성하고 파일로 저장 (미리보기)
        #     prompt_package = self._format_batch_prompt(batch, file_id_to_path_map)
        #     prompt_preview_path = prompts_dir / f"{batch['batch_id']}.json"
        #     utils.save_json(prompt_package, prompt_preview_path)

        #     # 5b: 실제 LLM API 호출 (테스트를 위해 우선 주석 처리)
        #     # print(f"Auditing batch {batch['batch_id']}...")
        #     # audit_results = self.llm_client.process_batch(
        #     #     [prompt_package['prompt']['user']], # 사용자 프롬프트는 하나로 합쳐짐
        #     #     prompt_package['prompt']['system']
        #     # )
        #     # result_path = results_dir / f"{batch['batch_id']}.json"
        #     # utils.save_json(audit_results, result_path)

        # # print("Step 5 Done: All audit results saved.")
        # print(
        #     "Step 5 Done: All prompt packages saved for review. LLM calls are currently disabled."
        # )

        # ------------------------------------------------------------
        # --- Step 6: 최종 리포트 생성 ---
        print("Step 6 Start: Generating final summary report...")
        # reporter = ReportGenerator(config.REPORTS_DIR)
        # reporter.generate_summary_report(
        #     job_id=job_id,
        #     project_db_path=project_db_path,
        #     results_dir=config.ARTIFACTS_DIR / "results",
        # )
        print("Step 6 Done: Final report generated.")

        print("\n" + "=" * 20 + " Audit End " + "=" * 22)
