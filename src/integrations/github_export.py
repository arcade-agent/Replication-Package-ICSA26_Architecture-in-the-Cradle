"""Read GitHub closing links into ArchGuard's issue/MR batch schema via gh.

No links are inferred from prose and no architectural-smell labels are generated.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

QUERY = """
query($owner:String!, $name:String!, $number:Int!) {
  repository(owner:$owner, name:$name) {
    pullRequest(number:$number) {
      closingIssuesReferences(first:100) {
        pageInfo { hasNextPage }
        nodes { number title body state createdAt updatedAt url
                repository { nameWithOwner } }
      }
    }
  }
}
"""


def obj(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        raise ValueError("Expected a JSON object with string keys")
    return dict(value)


def items(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError("Expected a JSON array")
    return list(value)


def string(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("Expected a nonempty string")
    return value


def number(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("Expected a positive issue/PR number")
    return value


def valid_repo(repo: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", repo):
        raise ValueError("Repository must be owner/name")
    return repo


class GitHubAPI(Protocol):
    def rest(self, path: str, *, paginated: bool = False) -> object: ...

    def closing_links(self, repo: str, pull_number: int) -> object: ...


class GhCLI:
    @staticmethod
    def _call(arguments: list[str]) -> object:
        result = subprocess.run(
            ["gh", "api", *arguments], check=True, capture_output=True, text=True, timeout=60
        )
        return json.loads(result.stdout)

    def rest(self, path: str, *, paginated: bool = False) -> object:
        return self._call([path, "--paginate", "--slurp"] if paginated else [path])

    def closing_links(self, repo: str, pull_number: int) -> object:
        owner, name = valid_repo(repo).split("/")
        return self._call(
            [
                "graphql",
                "-f",
                f"query={QUERY}",
                "-f",
                f"owner={owner}",
                "-f",
                f"name={name}",
                "-F",
                f"number={pull_number}",
            ]
        )


def linked_issues(payload: object) -> list[dict[str, object]]:
    root = obj(payload)
    if root.get("errors"):
        raise ValueError("GraphQL errors; refusing incomplete export")
    repository = obj(obj(root["data"])["repository"])
    links = obj(obj(repository["pullRequest"])["closingIssuesReferences"])
    if obj(links["pageInfo"])["hasNextPage"] is not False:
        raise ValueError("More than 100 closing links; refusing a truncated export")
    return [obj(issue) for issue in items(links["nodes"])]


def commits_for_pull(api: GitHubAPI, repo: str, pull_number: int) -> list[dict[str, object]]:
    commits = [
        obj(commit)
        for page in items(
            api.rest(
                f"repos/{repo}/pulls/{pull_number}/commits",
                paginated=True,
            )
        )
        for commit in items(page)
    ]
    if not commits or len(commits) >= 250:
        raise ValueError("Empty or potentially truncated PR commit list")
    output: list[dict[str, object]] = []
    for commit in commits:
        sha = string(commit["sha"])
        if not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise ValueError("Expected a full commit SHA")
        pages = [
            obj(page)
            for page in items(
                api.rest(
                    f"repos/{repo}/commits/{sha}",
                    paginated=True,
                )
            )
        ]
        if not pages or any(page.get("sha") != sha for page in pages):
            raise ValueError("Commit response did not match requested SHA")
        files = [obj(file) for page in pages for file in items(page["files"])]
        if len(files) >= 3000:
            raise ValueError("Potentially truncated commit file list")
        metadata = obj(pages[0]["commit"])
        output.append(
            {
                "commit_metadata": {
                    "id": sha,
                    "authored_date": string(obj(metadata["author"])["date"]),
                    "committed_date": string(obj(metadata["committer"])["date"]),
                },
                "commit_diff_summary": {
                    "files_summary": [
                        {
                            "new_path": string(file["filename"]),
                            "old_path": string(file.get("previous_filename", file["filename"])),
                            "status": string(file["status"]),
                        }
                        for file in files
                    ]
                },
            }
        )
    return output


def export(repo: str, pull_numbers: list[int], api: GitHubAPI) -> dict[str, object]:
    repo = valid_repo(repo)
    issues: dict[int, dict[str, object]] = {}
    relations: dict[int, list[dict[str, object]]] = {}
    skipped: list[dict[str, object]] = []
    unlinked: list[int] = []
    for pull_number in sorted(set(pull_numbers)):
        number(pull_number)
        pull = obj(api.rest(f"repos/{repo}/pulls/{pull_number}"))
        if pull.get("number") != pull_number or not pull.get("merged_at"):
            raise ValueError(f"PR #{pull_number} must be merged and match requested number")
        own_links: list[dict[str, object]] = []
        for issue in linked_issues(api.closing_links(repo, pull_number)):
            issue_repo = string(obj(issue["repository"])["nameWithOwner"])
            if issue_repo.lower() != repo.lower():
                skipped.append(
                    {
                        "repo": issue_repo,
                        "issue": number(issue["number"]),
                        "pull_request": pull_number,
                    }
                )
            else:
                own_links.append(issue)
        if not own_links:
            unlinked.append(pull_number)
            continue
        mr: dict[str, object] = {
            "mr_metadata": {
                "iid": pull_number,
                "title": string(pull["title"]),
                "merged_at": string(pull["merged_at"]),
                "merge_status": "merged",
                "source_branch": string(obj(pull["head"])["ref"]),
                "web_url": string(pull["html_url"]),
            },
            "commits": commits_for_pull(api, repo, pull_number),
            "provenance": {
                "provider": "github",
                "repository": repo,
                "relation": "closingIssuesReferences",
            },
        }
        for issue in own_links:
            iid = number(issue["number"])
            issues[iid] = issue
            relations.setdefault(iid, []).append(mr)
    records: list[dict[str, object]] = []
    for iid, issue in sorted(issues.items()):
        body = issue.get("body")
        if body is not None and not isinstance(body, str):
            raise ValueError("Issue body must be string or null")
        records.append(
            {
                "issue_data": {
                    "issue_metadata": {
                        "iid": iid,
                        "id": iid,
                        "title": string(issue["title"]),
                        "description": body or "",
                        "state": string(issue["state"]).lower(),
                        "created_at": string(issue["createdAt"]),
                        "updated_at": string(issue["updatedAt"]),
                        "web_url": string(issue["url"]),
                    }
                },
                "related_mrs": relations[iid],
                "summary": {
                    "issue_id": iid,
                    "issue_title": issue["title"],
                    "total_related_mrs": len(relations[iid]),
                },
            }
        )
    return {
        "issues": records,
        "metadata": {
            "provider": "github",
            "repository": repo,
            "total_issues": len(records),
            "pull_requests": sorted(set(pull_numbers)),
            "unlinked_pulls": unlinked,
            "skipped_cross_repository_links": skipped,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "relation_policy": "GitHub closingIssuesReferences only; no text inference",
            "discussion_policy": "not collected",
            "smell_labels": "not generated",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--pull-requests", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.pull_requests) > 20:
        parser.error("Export at most 20 explicit PRs per invocation")
    try:
        result = export(args.repo, args.pull_requests, GhCLI())
    except (
        ValueError,
        KeyError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as error:
        parser.exit(1, f"Export failed: {error}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
