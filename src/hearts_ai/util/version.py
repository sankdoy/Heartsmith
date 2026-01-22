from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version


def get_version_string() -> str:
    try:
        return version("hearts-ai")
    except PackageNotFoundError:
        return "unknown"
