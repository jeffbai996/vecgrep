"""Bounded, immutable review context captured from the exact mutation inputs."""
from __future__ import annotations

from difflib import unified_diff
import re


def change_diff(before: str, after: str, *, before_offset: int = 0, after_offset: int = 0, limit: int = 24000) -> dict:
    lines=list(unified_diff(before.splitlines(keepends=True),after.splitlines(keepends=True),fromfile='before',tofile='after',n=3))
    # Normalize a missing final newline so clients can parse the last +/- row.
    lines=[line if line.endswith('\n') else line+'\n' for line in lines]
    def shift_hunk(match: re.Match) -> str:
        return f"@@ -{int(match[1])+before_offset}{match[2] or ''} +{int(match[3])+after_offset}{match[4] or ''} @@"
    lines=[re.sub(r'^@@ -(\d+)(,\d+)? \+(\d+)(,\d+)? @@',shift_hunk,line) if line.startswith('@@ ') else line for line in lines]
    parts=[];size=0;truncated=False
    for line in lines:
        if size+len(line)>limit:
            truncated=True;break
        parts.append(line);size+=len(line)
    result=''.join(parts)
    if not any(line[:1] in ('+','-') and not line.startswith(('+++','---')) for line in parts): result=''
    return {'unified_diff':result,'diff_truncated':truncated}
