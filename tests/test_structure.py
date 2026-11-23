"""Fast checks that need no API keys, models or network."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_every_module_parses():
    for path in ROOT.rglob("*.py"):
        if ".venv" not in path.parts:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_importing_package_does_not_open_qdrant():
    tree = ast.parse((ROOT / "langchain_agent" / "__init__.py").read_text(encoding="utf-8"))
    imported = [n.module for n in tree.body if isinstance(n, ast.ImportFrom)]
    assert "langchain_agent.tools" not in imported
