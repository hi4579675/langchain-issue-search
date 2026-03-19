"""
수집 데이터 스키마
Java의 record / DTO 와 동일한 역할
"""
from dataclasses import dataclass, field
import datetime


@dataclass
class RawIssue:
    id: int
    number: int
    title: str
    body: str
    created_at: datetime.datetime
    labels: list[str] = field(default_factory=list)


@dataclass
class RawComment:
    id: int
    issue_number: int
    body: str
    created_at: datetime.datetime
    reactions: int = 0          # 👍 수 → solution 판별에 활용


@dataclass
class CleanedIssue:
    """정제 완료된 Issue-Solution 쌍"""
    issue_number: int
    title: str
    question: str               # 정제된 이슈 본문
    created_at: datetime.datetime
    solution: str | None = None  # 채택된 댓글 (없을 수 있음)
    labels: list[str] = field(default_factory=list)
    repo: str = "langchain-ai/langchain"
