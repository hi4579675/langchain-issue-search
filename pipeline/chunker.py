"""텍스트 청킹 — 코드블록과 일반 텍스트 분리"""
import re
from dataclasses import dataclass, field

MAX_TEXT_LEN = 1000
CODE_BLOCK_RE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)

@dataclass
class Chunk:
    content: str
    chunk_type: str  # text | code
    language: str = ""
    metadata: dict = field(default_factory=dict)

def split_into_chunks(text: str, issue_number: int) -> list[Chunk]:
    chunks, last_end = [], 0
    for match in CODE_BLOCK_RE.finditer(text):
        before = text[last_end:match.start()].strip()
        if before:
            chunks.extend(_split_text(before, issue_number))
        code = match.group(2).strip()
        if code:
            chunks.append(Chunk(content=code, chunk_type="code",
                                language=match.group(1) or "unknown",
                                metadata={"issue_number": issue_number, "weight": 1.5}))
        last_end = match.end()
    tail = text[last_end:].strip()
    if tail:
        chunks.extend(_split_text(tail, issue_number))
    return chunks

def _split_text(text: str, issue_number: int) -> list[Chunk]:
    return [Chunk(content=text[i:i+MAX_TEXT_LEN].strip(), chunk_type="text",
                  metadata={"issue_number": issue_number, "weight": 1.0})
            for i in range(0, len(text), MAX_TEXT_LEN) if text[i:i+MAX_TEXT_LEN].strip()]
