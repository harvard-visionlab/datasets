"""Every ``visionlab-datasets`` command shown in the docs must parse with the real CLI.

Covers the README's code blocks, the skill (skills/visionlab-datasets/SKILL.md), the CLI module docstring and the
parser's own examples, so documented syntax can't drift from the code.
"""
import re
import shlex
from pathlib import Path

import pytest

import visionlab.datasets.cli as cli

ROOT = Path(__file__).resolve().parents[1]
DOCS = [ROOT / "README.md", ROOT / "skills" / "visionlab-datasets" / "SKILL.md"]


def _code_blocks(text: str) -> list[str]:
    return re.findall(r"```[a-z]*\n(.*?)```", text, flags=re.S)


def _commands(text: str) -> list[str]:
    """``visionlab-datasets ...`` invocations (args only), backslash continuations joined, comments dropped."""
    out = []
    for block in text if isinstance(text, list) else [text]:
        joined = re.sub(r"\\\n\s*", " ", block)
        for line in joined.splitlines():
            line = line.split(" #", 1)[0].strip()
            m = re.search(r"(?:^|uv run (?:--no-sync )?|/bin/)visionlab-datasets\s+(.*)$", line)
            if not m or m.group(1).startswith("=") or "..." in m.group(1) or "<" in m.group(1):
                continue  # not a command, or a placeholder like `visionlab-datasets ...`
            out.append(m.group(1).strip())
    return out


def _documented() -> list[tuple[str, str]]:
    found = []
    for doc in DOCS:
        found += [(doc.name, c) for c in _commands(_code_blocks(doc.read_text()))]
    found += [("cli docstring", c) for c in _commands(cli.__doc__)]
    found += [("parser epilog", c) for c in _commands(cli.build_parser().epilog)]
    return found


def test_docs_have_commands():
    by_doc = {d for d, _ in _documented()}
    assert {"README.md", "SKILL.md", "cli docstring", "parser epilog"} <= by_doc


@pytest.mark.parametrize("where, command", _documented(), ids=lambda v: v if isinstance(v, str) else None)
def test_documented_command_parses(where, command):
    args = cli.build_parser().parse_args(shlex.split(command))
    if getattr(args, "dataset", None):
        cli.resolve_name(args.dataset)
    for name in getattr(args, "datasets", None) or []:
        cli.resolve_name(name)
    if args.command in ("path", "sync"):
        if args.splits is not None:
            cli.parse_list(args.splits, cli.SPLITS, list(cli.SPLITS))
        fmt = cli.fmt_arg(args)
        assert fmt is not None or args.command == "path", f"{where}: sync without a format: {command}"
        cli.parse_list(fmt, cli.FMTS, list(cli.FMTS_ALL), cli.FMTS_ALL)
