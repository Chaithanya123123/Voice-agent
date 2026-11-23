"""Support agent package.

The agent is loaded on first use. Importing `langchain_agent.rag` from the API must not
open the local Qdrant store a second time, because it allows only one client at once.
"""
from importlib import import_module

__all__ = ["nano_agent", "lauki_agent", "qdrant", "embedder"]


def __getattr__(name: str):
    if name in __all__:
        return getattr(import_module("langchain_agent.tools"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
