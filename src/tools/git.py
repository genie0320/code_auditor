# src/tools/git.py

"""Git 히스토리를 조회하는 도구들"""
import subprocess


# (수정) 'directory' 인자 추가
def search_git_log(directory: str, days: int, query: str | None = None) -> str:
    """최근 N일 동안의 git log를 조회합니다. 특정 쿼리가 있으면 커밋 메시지를 필터링합니다."""
    print(
        f"--- Tool Called: search_git_log(directory='{directory}', days={days}, query='{query}') ---"
    )
    try:
        command = ["git", "log", f"--since={days}.days.ago"]
        if query:
            command.extend(["--grep", query, "-i"])

        # (수정) cwd 파라미터를 사용하여 git 명령어를 실행할 디렉토리 지정
        result = subprocess.run(
            command, capture_output=True, text=True, check=True, cwd=directory
        )
        return result.stdout
    except FileNotFoundError:
        return "Error: 'git' command not found. Is Git installed and in your PATH?"
    except subprocess.CalledProcessError as e:
        return f"Error running git log: {e.stderr}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"
