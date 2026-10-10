"""Restricted terminal and code-writing tools for the chatbot agent."""
from __future__ import annotations

import ast
import shlex
import subprocess
from pathlib import Path

from langchain_core.tools import tool

from config.settings import AGENT_WORKSPACE, SANDBOX_COMMAND_TIMEOUT_SECONDS


_READ_ONLY_COMMANDS = {
    "cat",
    "find",
    "git",
    "grep",
    "head",
    "ls",
    "pip",
    "pwd",
    "pytest",
    "python",
    "rg",
    "tail",
    "wc",
}
_GIT_READ_ONLY_SUBCOMMANDS = {"branch", "diff", "log", "status", "show"}
_FORBIDDEN_TOKENS = (";", "|", "&", ">", "<", "`", "$(", "\n", "\r")
_FORBIDDEN_ARGUMENTS = {"-delete", "-exec", "-execdir", "-ok", "-okdir"}


def _workspace_root() -> Path:
    """Return the resolved workspace root used for sandbox validation."""
    return Path(AGENT_WORKSPACE).resolve()


def _within_workspace(path: Path, workspace: Path) -> bool:
    """Return whether a resolved path is inside the workspace directory."""
    try:
        path.relative_to(workspace)
        return True
    except ValueError:
        return False


def _has_outside_path_argument(args: list[str]) -> bool:
    """Return whether command arguments attempt to address an outside path."""
    for arg in args[1:]:
        normalized = arg.replace("\\", "/")
        if normalized.startswith("/") or normalized.startswith("../"):
            return True
        if any(part == ".." for part in normalized.split("/")):
            return True
        if len(normalized) >= 3 and normalized[1:3] == ":/":
            return True
    return False


@tool
def shell_tool(command: str) -> str:
    """Run one allowlisted, read-only command inside the agent workspace.

    Shell operators, interpreters, package installation, and destructive
    commands are rejected. Use this for inspection, tests, and diagnostics.
    """
    if not command.strip():
        return "Validation Failed: command must not be empty."
    if any(token in command for token in _FORBIDDEN_TOKENS):
        return "Security Error: shell operators and multiline commands are not allowed."

    try:
        args = shlex.split(command, posix=True)
    except ValueError as exc:
        return f"Validation Failed: invalid command syntax: {exc}"
    if not args or args[0] not in _READ_ONLY_COMMANDS:
        return "Security Error: command is not in the read-only allowlist."
    if _has_outside_path_argument(args):
        return "Security Error: command paths must remain inside the agent workspace."
    if any(arg in _FORBIDDEN_ARGUMENTS for arg in args):
        return "Security Error: command execution arguments are not allowed."
    if args[0] == "git" and (len(args) < 2 or args[1] not in _GIT_READ_ONLY_SUBCOMMANDS):
        return "Security Error: only read-only git subcommands are allowed."
    if args[0] == "pip" and (len(args) < 2 or args[1] not in {"list", "show"}):
        return "Security Error: only pip list and pip show are allowed."
    if args[0] == "python" and not any(arg in {"--version", "-V"} for arg in args[1:]):
        return "Security Error: Python execution is limited to version checks."

    workspace = _workspace_root()
    workspace.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            ["bash", "-lc", command],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=SANDBOX_COMMAND_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"Error executing command: {exc}"

    output = (result.stdout + result.stderr).strip()
    if result.returncode:
        return f"Command failed with exit code {result.returncode}:\n{output}"
    return output or "Command completed successfully with no output."


@tool
def safe_write_code(filepath: str, content: str) -> str:
    """Create or update a file under the dedicated agent workspace.

    Python files are parsed with AST validation before they are written.
    """
    workspace = _workspace_root()
    target_path = (workspace / filepath).resolve()
    if not _within_workspace(target_path, workspace):
        return "Security Error: modifications are restricted to the agent workspace."
    if target_path.exists() and target_path.is_symlink():
        return "Security Error: symlink targets cannot be modified."

    if target_path.suffix.lower() == ".py":
        try:
            ast.parse(content, filename=str(target_path))
        except SyntaxError as exc:
            line = exc.lineno or 0
            return (
                f"Validation Failed: syntax error detected on line {line}: "
                f"{exc.msg}. File was not written."
            )

    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(content, encoding="utf-8")
    except OSError as exc:
        return f"Error modifying file: {exc}"
    return f"Success: {filepath} validated and written to the agent workspace."
