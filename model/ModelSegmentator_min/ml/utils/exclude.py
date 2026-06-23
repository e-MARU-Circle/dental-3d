from __future__ import annotations

from pathlib import Path
from typing import Iterable, Set


def read_exclude_list(path: str | Path | None) -> Set[str]:
    """Return a set of case ids to exclude (stem names without extension).

    Lines starting with '#' or blank lines are ignored. If path is None or the
    file does not exist, an empty set is returned.
    """
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        return set()
    items: Set[str] = set()
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        items.add(line)
    return items


def filter_paths(paths: Iterable[Path], exclude: Iterable[str]) -> list[Path]:
    """Filter out paths whose stem matches any entry in *exclude*."""
    exclude_set = set(exclude)
    if not exclude_set:
        return list(paths)
    return [p for p in paths if p.stem not in exclude_set]
