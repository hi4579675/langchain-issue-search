"""
GitHub API 클라이언트
Rate limit: 5,000 req/h (인증 시)
Java의 HttpClient wrapper 클래스와 동일한 역할
"""
import time
import requests
from dataclasses import dataclass, field


@dataclass
class GitHubClient:
    token: str
    base_url: str = "https://api.github.com"
    _remaining: int = field(default=5000, repr=False, init=False)
    _reset_at: float = field(default=0.0, repr=False, init=False)

    def get(self, path: str, params: dict | None = None) -> dict | list:
        self._wait_if_needed()
        resp = requests.get(
            f"{self.base_url}{path}",
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
            },
            params=params or {},
            timeout=10,
        )
        self._update_rate_limit(resp)
        resp.raise_for_status()
        return resp.json()

    def _wait_if_needed(self):
        if self._remaining < 10:
            wait = max(0, self._reset_at - time.time()) + 1
            print(f"Rate limit 임박 — {wait:.0f}초 대기")
            time.sleep(wait)

    def get_paginated(self, path: str, params: dict | None = None):
        """Link 헤더를 따라 전체 페이지를 순회하는 제너레이터"""
        params = dict(params or {})
        params.setdefault("per_page", 100)
        url = f"{self.base_url}{path}"
        while url:
            self._wait_if_needed()
            resp = requests.get(
                url,
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Accept": "application/vnd.github+json",
                },
                params=params,
                timeout=10,
            )
            self._update_rate_limit(resp)
            resp.raise_for_status()
            yield from resp.json()
            url = self._parse_next_link(resp.headers.get("Link", ""))
            params = {}  # 다음 페이지 URL에 이미 파라미터가 포함됨

    def _parse_next_link(self, link_header: str) -> str | None:
        """Link: <url>; rel="next" 형식에서 다음 페이지 URL 추출"""
        for part in link_header.split(","):
            if 'rel="next"' in part:
                return part.split(";")[0].strip().strip("<>")
        return None

    def _update_rate_limit(self, resp: requests.Response):
        self._remaining = int(resp.headers.get("X-RateLimit-Remaining", 5000))
        self._reset_at  = float(resp.headers.get("X-RateLimit-Reset", 0))
