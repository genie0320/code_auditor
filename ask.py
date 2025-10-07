# src/ask.py

from langchain.agents import AgentExecutor, create_openai_tools_agent, Tool
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.callbacks import FileCallbackHandler

import sys
import config
import functools
from pathlib import Path
from datetime import datetime


from src.tools import file_system, git
from src.utils.spec_generator import create_spec_file


def main():
    """
    Main function to execute the code analysis and question answering process.

    Returns:
        _type_: _description_
    """
    # --- 1. 환경 설정 및 경로 확인 ---
    target_dir = Path(config.TARGET_ROOT_DIR)
    if not target_dir.is_dir():
        print(f"⚠️ 경고: 설정된 타겟 폴더를 찾을 수 없습니다: {target_dir.resolve()}")
        return
    print(f"✅ 타겟 폴더를 확인했습니다: {target_dir.resolve()}")

    # --- 2. 프로젝트 명세서 준비 ---
    spec_path = target_dir / "project_spec.md"
    if not spec_path.exists():
        print(f"프로젝트 명세를 작성하는 중... (대상: {target_dir})")
        project_spec_content = create_spec_file(target_dir)
    else:
        print(f"Loading existing project_spec.md from {target_dir}...")
        project_spec_content = spec_path.read_text(encoding="utf-8")

    # --- 3. 사용자 질문 입력 ---
    question = input("\n무엇을 도와드릴까요?: ")
    if not question:
        return
    print(f"\nQuestion: {question}\n")

    def find_similar_code(code_snippet: str, k: int = 5) -> list[dict]:
        """주어진 코드 조각과 가장 유사한 코드 청크 k개를 벡터 DB에서 찾아 반환합니다."""
        print(f"--- Tool Called: find_similar_code(k={k}) ---")
        embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
        vector_store = Chroma(
            persist_directory=config.CHROMA_DB_PATH, embedding_function=embeddings
        )
        results = vector_store.similarity_search_with_score(code_snippet, k=k)
        return [
            {"score": score, "content": doc.page_content, "metadata": doc.metadata}
            for doc, score in results
        ]

    # TODO: 정말 강력하게 별도 파일로 분리하고 싶다.
    tools = [
        Tool(
            name="file_system_search",
            func=lambda pattern: file_system.search_files(
                directory=str(target_dir), pattern=pattern
            ),
            description="Use this to search for files in the target codebase using a glob pattern like '**/*.js'.",
        ),
        Tool(
            name="read_file_content",
            func=lambda path: file_system.read_file_content(
                base_dir=str(target_dir), file_path=path
            ),
            description="Use this to read the full content of a specific file. Input is a file path relative to the project root.",
        ),
        Tool(
            name="git_history_search",
            func=lambda q: git.search_git_log(
                directory=str(target_dir), days=30, query=q
            ),
            description="Use this to search git commit messages within the last 30 days. Input is an optional query string.",
        ),
        Tool(
            name="similar_code_search",
            func=find_similar_code,
            description="Use this to find code chunks that are semantically similar to a given code snippet.",
        ),
    ]
    print("--- Step: Initialized Tools ---")

    # --- 5. 프롬프트 생성 ---
    # TODO: 정말 강력하게 별도 파일로 분리하고 싶다.
    system_prompt = f"""
You are an expert Senior Developer's assistant. Your goal is to answer the user's question about a codebase by thinking step-by-step. First, analyze the user's non-technical question and translate it into a technical plan, using the provided project specification. Then, execute that plan using your available tools.

--- Project Specification ---
{project_spec_content}
---
"""
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    print("--- Step: Created Meta-Prompt for Tool Calling Agent ---")

    # --- 6. 에이전트 생성 및 실행 ---
    print("--- Step: Initializing and Running Agent ---")
    result = None
    try:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        logfile_path = (
            log_dir / f"agent_trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

        with FileCallbackHandler(str(logfile_path)) as handler:
            llm = ChatOpenAI(
                temperature=0,
                model=config.LLM_DEFAULT_MODEL,
                api_key=config.LLM_API_KEYS.get("openai"),
            )
            agent = create_openai_tools_agent(llm, tools, prompt_template)
            agent_executor = AgentExecutor(
                agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
            )

            result = agent_executor.invoke(
                {"input": question}, config={"callbacks": [handler]}
            )
    except Exception as e:
        result = {"output": str(e)}

    # --- 7. 최종 결과 출력 ---
    print("\n--- Final Answer ---")
    print(result.get("output") if result else "No result obtained.")


if __name__ == "__main__":
    main()
