# src/utils/generate_spec.py

"""
코드베이스를 분석하여 project_spec.md 파일의 초안을 생성.
작업ID당 한번만 실행됨.
"""
import os
import json
from pathlib import Path
from collections import Counter

# 참고: LLM 요약을 위해서는 Orchestrator와 유사한 LLM 클라이언트 초기화가 필요합니다.
# 이 부분은 간소화를 위해 우선 플레이스홀더로 남겨두겠습니다.


def _detect_tech_stack(target_dir) -> str:
    print("... Analyzing dependency files...")
    """의존성 파일을 분석하여 기술 스택을 추출합니다."""
    tech = []
    if Path(target_dir / "requirements.txt").exists():
        tech.append("Python (requirements.txt)")
    if Path(target_dir / "package.json").exists():
        tech.append("JavaScript/TypeScript (package.json)")
    if Path(target_dir / "pom.xml").exists():
        tech.append("Java (Maven)")
    return "\n- ".join(tech) if tech else "Detection failed."


# TODO: 그냥 폴더와 포함된 파일명을 구분해서 보여주자. 폴더는 / 붙이고, 파일은 그냥.
def _analyze_directory_structure(target_dir) -> str:
    print("... Analyzing directory structure...")
    """주요 폴더를 스캔하여 프로젝트 구조를 파악합니다."""
    common_dirs = [
        "src",
        "app",
        "lib",
        "tests",
        "spec",
        "migrations",
        "entities",
        "controllers",
        "services",
        "routes",
        "views",
        "utils",
    ]
    found_dirs = set()
    for root, dirs, _ in os.walk(target_dir):
        # .git 등 숨김 폴더 제외
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for d in dirs:
            if d in common_dirs:
                found_dirs.add(f"`{d}/`")
    return (
        "\n- ".join(sorted(list(found_dirs)))
        if found_dirs
        else "No common directories found."
    )


def _get_language_composition(target_dir: str) -> str:
    print("--- Analyzing file extensions... ---")
    """파일 확장자를 분석하여 언어 구성을 계산합니다."""
    lang_counter = Counter()
    total_files = 0

    # 이 부분을 수정합니다: "." 대신 target_dir 사용
    for root, dirs, files in os.walk(target_dir):
        # 숨김 파일 및 폴더 제외 ('.')는 그대로 유지
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for file in files:
            if "." in file:
                ext = f".{file.split('.')[-1]}"
                lang_counter[ext] += 1
                total_files += 1

    if total_files == 0:
        # 특정 경로에 파일이 없을 경우 target_dir을 포함하여 메시지 출력
        return f"No files found in {target_dir}."

    composition_str = []
    for ext, count in lang_counter.most_common(5):
        percentage = (count / total_files) * 100
        composition_str.append(f"{ext}: {percentage:.1f}%")
    return "\n- ".join(composition_str)


def _summarize_readme_with_llm(target_dir) -> str:
    print("--- Summarizing README.md with LLM... ---")
    """README.md 파일 내용을 LLM으로 요약합니다."""
    if not Path(target_dir / "README.md").exists():
        return "README.md not found."
    # Placeholder: 실제로는 README.md 파일을 LLM에 보내 요약 요청
    return "This project is a RAG-based code audit service..."


# TODO: 제발 프롬프트들은 한곳으로 옮기자... 너무 흩어져있고 시야가 복잡해 죽을 것 같음.
def create_spec_file(target_dir) -> str:
    """프로젝트 명세서 파일을 생성하고, 그 내용을 문자열로 반환합니다."""
    print(f"Generating project_spec.md in {target_dir}...")

    spec_content = f"""# Auto-Generated Project Specification

## Project Summary
{_summarize_readme_with_llm(target_dir)}

## Language Composition
{_get_language_composition(target_dir)}

## Tech Stack
{_detect_tech_stack(target_dir)}

## Key Directory Structure
{_analyze_directory_structure(target_dir)}
"""

    with open(target_dir / "project_spec.md", "w", encoding="utf-8") as f:
        f.write(spec_content)

    print(f"Successfully generated project_spec.md in {target_dir}!")
    return spec_content
