"""
노이즈 제거 + solution 댓글 선별
판별 기준: reactions 수 가장 높은 댓글 = solution
"""
import re
from .schema import RawIssue, RawComment, CleanedIssue

MIN_BODY_LEN = 50
MIN_SOLUTION_LEN = 30

_NOISE_PATTERNS = [
    r"^(thanks?|thank you|thx|great|awesome|lgtm)[!.\s]*$",
    r"^\+1$",
    r"^👍+$",
]


def is_noise(text: str) -> bool:
    t = text.strip().lower()
    return any(re.match(p, t) for p in _NOISE_PATTERNS)


def pick_solution(comments: list[RawComment]) -> RawComment | None:
    valid = [c for c in comments
             if not is_noise(c.body) and len(c.body) >= MIN_SOLUTION_LEN]
    if not valid:
        return None
    return max(valid, key=lambda c: c.reactions)


def clean_text(text: str) -> str:
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)   # 이미지 제거
    text = re.sub(r"\[.*?\]\(.*?\)", "", text)     # 링크 제거
    text = re.sub(r"\r\n", "\n", text)             # Windows 줄바꿈 통일
    text = re.sub(r"\n{3,}", "\n\n", text)         # 연속 빈 줄 압축
    return text.strip()


def make_cleaned_issue(issue: RawIssue,
                       comments: list[RawComment]) -> CleanedIssue | None:
    if len(issue.body) < MIN_BODY_LEN:
        return None
    solution = pick_solution(comments)
    if solution is None:
        return None
    return CleanedIssue(
        issue_number=issue.number,
        title=issue.title,
        question=clean_text(issue.body),
        solution=clean_text(solution.body),
        labels=issue.labels,
        created_at=issue.created_at,
    )
