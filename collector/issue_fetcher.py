"""
LangChain GitHub Issue + Comment 수집
대상: langchain-ai/langchain, label=bug, state=closed
"""
import itertools
from datetime import datetime
from typing import Generator
from .github_client import GitHubClient
from .schema import RawIssue, RawComment

REPO = "langchain-ai/langchain"


def fetch_issues(client: GitHubClient,
                 label: str = "bug",
                 max_pages: int = 50) -> Generator[RawIssue, None, None]:
    """
    closed bug 이슈를 페이지 단위로 가져옴
    Java의 Iterator<RawIssue> 와 동일한 개념
    """
    pages = client.get_paginated(
        f"/repos/{REPO}/issues",
        params={"state": "closed", "labels": label},
    )
    for item in itertools.islice(pages, max_pages * 100):
        if "pull_request" in item:      # PR 제외
            continue
        yield RawIssue(
            id=item["id"],
            number=item["number"],
            title=item["title"],
            body=item.get("body") or "",
            created_at=datetime.fromisoformat(item["created_at"].replace("Z", "+00:00")),
            labels=[lb["name"] for lb in item["labels"]],
        )


def fetch_comments(client: GitHubClient,
                   issue_number: int) -> list[RawComment]:
    items = client.get(f"/repos/{REPO}/issues/{issue_number}/comments")
    return [
        RawComment(
            id=c["id"],
            issue_number=issue_number,
            body=c.get("body") or "",
            created_at=datetime.fromisoformat(c["created_at"].replace("Z", "+00:00")),
            reactions=c.get("reactions", {}).get("total_count", 0),
        )
        for c in items
    ]
