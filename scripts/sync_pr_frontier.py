#!/usr/bin/env python3
"""Sync and rank GitHub PRs for local frontier tracking.

This script uses the GitHub CLI (`gh`) as the transport layer, stores a
normalized JSON cache keyed by PR number, and writes a derived CSV sorted by a
user-selected ranking mode.

Typical usage:

  python3 scripts/sync_pr_frontier.py sync
  python3 scripts/sync_pr_frontier.py sync --sort bpb
  python3 scripts/sync_pr_frontier.py render --sort relevance
  python3 scripts/sync_pr_frontier.py render --filter clean
  python3 scripts/sync_pr_frontier.py render --include-closed --sort bpb

The cache is incremental:
- the REST API `/pulls` endpoint is used to page through all PRs (index fetch)
- the body/title are already included in the index response, so no secondary
  per-PR fetch is needed; only PRs that are missing or whose `updatedAt`
  changed are re-processed
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "openai/parameter-golf"
DEFAULT_CACHE_DIR = REPO_ROOT / "artifacts" / "gh_frontier"
SCHEMA_VERSION = 1

DEFAULT_RELEVANCE_WEIGHTS: dict[str, int] = {
    "sp8192": 8,
    "legal": 5,
    "score-first": 5,
    "score first": 5,
    "ttt": 5,
    "parallel residual": 4,
    "parallel residuals": 4,
    "loop_start=3": 4,
    "loop start": 2,
    "ngram": 3,
    "causal": 3,
    "normalized": 2,
    "faithful": 2,
    "runpod": 1,
    "h100": 1,
    "slot": -6,
    "leakage": -4,
}

FLAG_PATTERNS: dict[str, tuple[str, ...]] = {
    "mentions_sp8192": ("sp8192",),
    "mentions_legal": ("legal",),
    "mentions_ttt": ("ttt",),
    "mentions_parallel_residual": ("parallel residual", "parallel residuals"),
    "mentions_loop": ("loop_start", "loop start", "loop_end"),
    "mentions_ngram": ("ngram",),
    "mentions_causal": ("causal",),
    "mentions_normalized": ("normalized",),
    "mentions_slot": ("slot",),
    "mentions_runpod": ("runpod",),
    "mentions_h100": ("h100", "8xh100"),
}

BPB_PATTERNS = (
    re.compile(r"\b(?:val_bpb|bpb)\s*[:=]?\s*(\d+\.\d+)\b", re.IGNORECASE),
    re.compile(r"\b(\d+\.\d+)\s*BPB\b", re.IGNORECASE),
)
NATS_PATTERNS = (
    re.compile(r"\b(?:val_loss|nats?)\s*[:=]?\s*(\d+\.\d+)\b", re.IGNORECASE),
    re.compile(r"\b(\d+\.\d+)\s*nats?\b", re.IGNORECASE),
)
SEED_PATTERNS = (
    re.compile(r"\b(\d+)[ -]?seed(?:\s+mean)?\b", re.IGNORECASE),
    re.compile(r"\bseed(?:s)?\s*[:=]?\s*(\d+)\b", re.IGNORECASE),
)
BYTES_PATTERNS = (
    re.compile(
        r"\b(?:bytes_total|artifact(?: size)?|submission size|bytes)\s*[:=]?\s*([0-9][0-9,]{4,})\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b([0-9][0-9,]{5,})\s*bytes\b", re.IGNORECASE),
)


class GhCommandError(RuntimeError):
    """Raised when a gh CLI command fails."""


@dataclass(frozen=True)
class OutputPaths:
    cache_json: Path
    default_ranked_csv: Path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def repo_slug(repo: str) -> str:
    return repo.replace("/", "_")


def output_paths(cache_dir: Path, repo: str) -> OutputPaths:
    slug = repo_slug(repo)
    return OutputPaths(
        cache_json=cache_dir / f"{slug}_pr_cache.json",
        default_ranked_csv=cache_dir / f"{slug}_pr_ranked.csv",
    )


def ranked_csv_path(cache_dir: Path, repo: str, sort_mode: str, filters: list[str]) -> Path:
    slug = repo_slug(repo)
    if sort_mode == "combined" and not filters:
        return cache_dir / f"{slug}_pr_ranked.csv"

    suffix_parts = [sort_mode, *filters] if filters else [sort_mode]
    suffix = "__".join(suffix_parts)
    return cache_dir / f"{slug}_pr_ranked__{suffix}.csv"


def run_gh(args: list[str]) -> str:
    cmd = ["gh", *args]
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.strip() or proc.stdout.strip() or "unknown gh error"
        raise GhCommandError(f"`{' '.join(cmd)}` failed:\n{stderr}")
    return proc.stdout


def ensure_gh_auth() -> None:
    proc = subprocess.run(
        ["gh", "auth", "status"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0:
        return
    message = proc.stderr.strip() or proc.stdout.strip() or "gh auth status failed"
    raise GhCommandError(
        "GitHub CLI is installed but authentication is not usable.\n"
        "Re-authenticate with `gh auth login -h github.com` and rerun.\n\n"
        f"{message}"
    )


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalize_author(value: Any) -> str | None:
    if isinstance(value, dict):
        return value.get("login")
    if isinstance(value, str):
        return value
    return None


def normalize_labels(value: Any) -> list[str]:
    if isinstance(value, list):
        labels: list[str] = []
        for item in value:
            if isinstance(item, dict) and isinstance(item.get("name"), str):
                labels.append(item["name"])
            elif isinstance(item, str):
                labels.append(item)
        return sorted(set(labels))
    if isinstance(value, dict) and isinstance(value.get("nodes"), list):
        return normalize_labels(value["nodes"])
    return []


def parse_numeric_matches(text: str, patterns: tuple[re.Pattern[str], ...]) -> list[float]:
    matches: list[float] = []
    for pattern in patterns:
        for raw in pattern.findall(text):
            try:
                matches.append(float(raw))
            except ValueError:
                continue
    return sorted(set(matches))


def parse_int_matches(text: str, patterns: tuple[re.Pattern[str], ...]) -> list[int]:
    matches: list[int] = []
    for pattern in patterns:
        for raw in pattern.findall(text):
            cleaned = raw.replace(",", "")
            try:
                matches.append(int(cleaned))
            except ValueError:
                continue
    return sorted(set(matches))


def filter_float_range(values: list[float], low: float, high: float) -> list[float]:
    return [value for value in values if low <= value <= high]


def filter_int_range(values: list[int], low: int, high: int) -> list[int]:
    return [value for value in values if low <= value <= high]


def parse_track(text: str) -> str | None:
    lower = text.lower()
    if "track_non_record_16mb" in lower:
        return "track_non_record_16mb"
    if "track_10min_16mb" in lower or "16mb" in lower:
        return "track_10min_16mb"
    return None


def parse_tokenizer_family(title: str, body: str) -> str | None:
    title_lower = title.lower()
    body_lower = body.lower()

    if "sp8192" in title_lower:
        return "sp8192"
    if "sp1024" in title_lower:
        return "sp1024"

    body_has_sp8192 = "sp8192" in body_lower
    body_has_sp1024 = "sp1024" in body_lower
    if body_has_sp8192 and not body_has_sp1024:
        return "sp8192"
    if body_has_sp1024 and not body_has_sp8192:
        return "sp1024"
    return None


def keyword_hits(text: str, weights: dict[str, int]) -> tuple[int, list[str]]:
    lower = text.lower()
    score = 0
    hits: list[str] = []
    for term, weight in weights.items():
        if term in lower:
            score += weight
            hits.append(term)
    return score, hits


def derive_metrics(pr: dict[str, Any]) -> dict[str, Any]:
    title = pr.get("title") or ""
    body = pr.get("body") or ""
    text = f"{title}\n{body}"
    lower = text.lower()

    bpb_matches = filter_float_range(parse_numeric_matches(text, BPB_PATTERNS), 0.7, 2.5)
    nats_matches = filter_float_range(parse_numeric_matches(text, NATS_PATTERNS), 0.7, 5.0)
    seed_matches = filter_int_range(parse_int_matches(text, SEED_PATTERNS), 1, 50)
    bytes_matches = filter_int_range(parse_int_matches(text, BYTES_PATTERNS), 1_000_000, 100_000_000)
    relevance_score, relevance_hits = keyword_hits(text, DEFAULT_RELEVANCE_WEIGHTS)

    flags = {
        name: any(token in lower for token in tokens)
        for name, tokens in FLAG_PATTERNS.items()
    }

    return {
        "best_bpb_heuristic": min(bpb_matches) if bpb_matches else None,
        "best_nats_heuristic": min(nats_matches) if nats_matches else None,
        "best_bytes_heuristic": min(bytes_matches) if bytes_matches else None,
        "seed_count_heuristic": max(seed_matches) if seed_matches else None,
        "bpb_matches": bpb_matches,
        "nats_matches": nats_matches,
        "bytes_matches": bytes_matches,
        "track_family": parse_track(text),
        "tokenizer_family": parse_tokenizer_family(title, body),
        "relevance_score": relevance_score,
        "relevance_hits": relevance_hits,
        **flags,
    }


def compact_pr_record(pr: dict[str, Any]) -> dict[str, Any]:
    base = {
        "number": pr.get("number"),
        "title": pr.get("title"),
        "url": pr.get("url"),
        "state": pr.get("state"),
        "is_draft": bool(pr.get("isDraft")),
        "author": normalize_author(pr.get("author")),
        "labels": normalize_labels(pr.get("labels")),
        "created_at": pr.get("createdAt"),
        "updated_at": pr.get("updatedAt"),
        "closed_at": pr.get("closedAt"),
        "merged_at": pr.get("mergedAt"),
        "review_decision": pr.get("reviewDecision"),
        "merge_state_status": pr.get("mergeStateStatus"),
        "changed_files": pr.get("changedFiles"),
        "additions": pr.get("additions"),
        "deletions": pr.get("deletions"),
        "body": pr.get("body") or "",
    }
    base.update(derive_metrics(pr))
    return base


def recompute_cached_record(record: dict[str, Any]) -> dict[str, Any]:
    updated = dict(record)
    updated.update(
        derive_metrics(
            {
                "title": record.get("title") or "",
                "body": record.get("body") or "",
            }
        )
    )
    return updated


def apply_filters(records: list[dict[str, Any]], filters: list[str]) -> list[dict[str, Any]]:
    filtered = records

    for filter_name in filters:
        if filter_name == "clean":
            filtered = [
                pr
                for pr in filtered
                if pr.get("best_bpb_heuristic") is not None
                and pr.get("tokenizer_family") == "sp8192"
                and not pr.get("mentions_slot")
            ]
            continue

        if filter_name == "sp8192":
            filtered = [pr for pr in filtered if pr.get("tokenizer_family") == "sp8192"]
            continue

        if filter_name == "open":
            filtered = [pr for pr in filtered if pr.get("state") == "open"]
            continue

        raise ValueError(f"Unknown filter: {filter_name}")

    return filtered


def effective_filters(filters: list[str], include_closed: bool) -> list[str]:
    final_filters = list(filters)
    if not include_closed and "open" not in final_filters:
        final_filters.append("open")
    return final_filters


def fetch_index(repo: str, limit: int, state: str) -> list[dict[str, Any]]:
    page = 1
    page_size = 100
    rows: list[dict[str, Any]] = []

    while len(rows) < limit:
        raw = run_gh(
            [
                "api",
                f"repos/{repo}/pulls?state={state}&per_page={page_size}&page={page}",
            ]
        )
        data = json.loads(raw)
        if not isinstance(data, list):
            raise GhCommandError("Unexpected JSON payload from `gh api repos/.../pulls`")
        if not data:
            break

        for item in data:
            if not isinstance(item, dict):
                continue
            labels = item.get("labels") if isinstance(item.get("labels"), list) else []
            rows.append(
                {
                    "additions": None,
                    "author": {"login": (item.get("user") or {}).get("login")},
                    "body": item.get("body") or "",
                    "changedFiles": None,
                    "closedAt": item.get("closed_at"),
                    "createdAt": item.get("created_at"),
                    "deletions": None,
                    "isDraft": bool(item.get("draft")),
                    "labels": labels,
                    "mergeStateStatus": None,
                    "mergedAt": item.get("merged_at"),
                    "number": item.get("number"),
                    "reviewDecision": None,
                    "state": item.get("state"),
                    "title": item.get("title"),
                    "updatedAt": item.get("updated_at"),
                    "url": item.get("html_url"),
                }
            )
            if len(rows) >= limit:
                break

        if len(data) < page_size:
            break
        page += 1

    return rows


def should_refresh(index_pr: dict[str, Any], existing: dict[str, Any] | None, force_all: bool) -> bool:
    if force_all or existing is None:
        return True
    return existing.get("updated_at") != index_pr.get("updatedAt")


def sorted_records(records: list[dict[str, Any]], sort_mode: str) -> list[dict[str, Any]]:
    def bpb_bucket(pr: dict[str, Any]) -> tuple[int, float]:
        value = pr.get("best_bpb_heuristic")
        return (0, float(value)) if value is not None else (1, float("inf"))

    def updated_bucket(pr: dict[str, Any]) -> str:
        return pr.get("updated_at") or ""

    if sort_mode == "bpb":
        return sorted(
            records,
            key=lambda pr: (
                *bpb_bucket(pr),
                -int(pr.get("relevance_score") or 0),
                -(pr.get("number") or 0),
            ),
        )
    if sort_mode == "relevance":
        return sorted(
            records,
            key=lambda pr: (
                -(pr.get("relevance_score") or 0),
                *bpb_bucket(pr),
                -(pr.get("number") or 0),
            ),
        )
    if sort_mode == "updated":
        return sorted(records, key=lambda pr: updated_bucket(pr), reverse=True)
    if sort_mode == "number":
        return sorted(records, key=lambda pr: pr.get("number") or 0, reverse=True)
    if sort_mode == "combined":
        return sorted(
            records,
            key=lambda pr: (
                -int(pr.get("relevance_score") or 0),
                *bpb_bucket(pr),
                -(pr.get("number") or 0),
            ),
        )
    raise ValueError(f"Unknown sort mode: {sort_mode}")


def write_ranked_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "number",
        "title",
        "state",
        "is_draft",
        "author",
        "created_at",
        "updated_at",
        "merged_at",
        "best_bpb_heuristic",
        "best_nats_heuristic",
        "seed_count_heuristic",
        "best_bytes_heuristic",
        "track_family",
        "tokenizer_family",
        "relevance_score",
        "relevance_hits",
        "labels",
        "mentions_sp8192",
        "mentions_legal",
        "mentions_ttt",
        "mentions_parallel_residual",
        "mentions_loop",
        "mentions_ngram",
        "mentions_causal",
        "mentions_normalized",
        "mentions_slot",
        "mentions_runpod",
        "mentions_h100",
        "review_decision",
        "merge_state_status",
        "changed_files",
        "additions",
        "deletions",
        "url",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for pr in records:
            row = dict(pr)
            row["labels"] = ",".join(pr.get("labels") or [])
            row["relevance_hits"] = ",".join(pr.get("relevance_hits") or [])
            writer.writerow({name: row.get(name) for name in fieldnames})


def sync_cache(
    repo: str,
    cache_json: Path,
    limit: int,
    state: str,
    force_all: bool,
) -> tuple[dict[str, Any], int]:
    ensure_gh_auth()
    existing = read_json(cache_json)
    cached_prs = existing.get("pull_requests", {}) if isinstance(existing, dict) else {}

    index_rows = fetch_index(repo=repo, limit=limit, state=state)
    refreshed = 0
    new_cache_prs: dict[str, Any] = dict(cached_prs)

    for index_pr in index_rows:
        number = index_pr.get("number")
        if not isinstance(number, int):
            continue
        key = str(number)
        if not should_refresh(index_pr, cached_prs.get(key), force_all=force_all):
            continue
        new_cache_prs[key] = compact_pr_record(index_pr)
        refreshed += 1

    payload = {
        "schema_version": SCHEMA_VERSION,
        "repo": repo,
        "updated_at": utc_now_iso(),
        "index_state": state,
        "index_limit": limit,
        "pull_requests": {
            key: new_cache_prs[key]
            for key in sorted(new_cache_prs, key=lambda item: int(item))
        },
    }
    write_json(cache_json, payload)
    return payload, refreshed


def render_from_cache(
    cache_json: Path,
    ranked_csv: Path,
    sort_mode: str,
    filters: list[str],
) -> tuple[int, int, list[dict[str, Any]]]:
    payload = read_json(cache_json)
    if not payload:
        raise SystemExit(f"Cache file does not exist yet: {cache_json}")
    pull_requests = payload.get("pull_requests", {})
    if not isinstance(pull_requests, dict):
        raise SystemExit(f"Cache file has unexpected shape: {cache_json}")
    records = [recompute_cached_record(value) for value in pull_requests.values() if isinstance(value, dict)]
    filtered_records = apply_filters(records, filters=filters)
    ordered = sorted_records(filtered_records, sort_mode=sort_mode)
    write_ranked_csv(ranked_csv, ordered)
    return len(records), len(filtered_records), ordered


def print_summary(records: list[dict[str, Any]], top_n: int) -> None:
    print(f"Top {min(top_n, len(records))} PRs:")
    for pr in records[:top_n]:
        number = pr.get("number")
        best_bpb = pr.get("best_bpb_heuristic")
        relevance = pr.get("relevance_score")
        state = pr.get("state")
        title = pr.get("title") or ""
        bpb_text = f"{best_bpb:.4f}" if isinstance(best_bpb, float) else "n/a  "
        print(f"#{number:<5} bpb={bpb_text:<12} rel={relevance:<3} state={state:<6} {title}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO, help=f"GitHub repo (default: {DEFAULT_REPO})")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Directory for cache JSON and derived CSV (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--sort",
        choices=("combined", "bpb", "relevance", "updated", "number"),
        default="combined",
        help="How to order the derived CSV/report",
    )
    parser.add_argument(
        "--filter",
        dest="filters",
        action="append",
        choices=("clean", "sp8192", "open"),
        default=[],
        help="Optional record filter. May be repeated.",
    )
    parser.add_argument(
        "--include-closed",
        action="store_true",
        help="Keep closed PRs in the rendered CSV. By default the CSV only includes open PRs.",
    )
    parser.add_argument("--top", type=int, default=15, help="How many ranked rows to print to stdout")

    subparsers = parser.add_subparsers(dest="command", required=True)

    sync_parser = subparsers.add_parser("sync", help="Refresh the JSON cache from gh and regenerate the CSV")
    sync_parser.add_argument("--limit", type=int, default=2000, help="Maximum number of PRs to index via gh pr list")
    sync_parser.add_argument(
        "--state",
        choices=("open", "closed", "all"),
        default="all",
        help="PR state filter for the index fetch (GitHub REST API supports: open, closed, all)",
    )
    sync_parser.add_argument(
        "--force-all",
        action="store_true",
        help="Re-fetch details for every indexed PR instead of only missing/updated ones",
    )

    subparsers.add_parser("render", help="Regenerate the ranked CSV from an existing JSON cache only")
    return parser


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    filters = effective_filters(args.filters, include_closed=args.include_closed)
    paths = output_paths(cache_dir=args.cache_dir, repo=args.repo)
    ranked_csv = ranked_csv_path(
        cache_dir=args.cache_dir,
        repo=args.repo,
        sort_mode=args.sort,
        filters=filters,
    )

    try:
        if args.command == "sync":
            payload, refreshed = sync_cache(
                repo=args.repo,
                cache_json=paths.cache_json,
                limit=args.limit,
                state=args.state,
                force_all=args.force_all,
            )
            total_count, filtered_count, ordered = render_from_cache(
                paths.cache_json,
                ranked_csv,
                sort_mode=args.sort,
                filters=filters,
            )
            print(f"Repo: {payload['repo']}")
            print(f"Cache: {paths.cache_json}")
            print(f"CSV:   {ranked_csv}")
            print(f"Cached PRs: {total_count}")
            print(f"Rows written: {filtered_count}")
            print(f"Refreshed PRs this run: {refreshed}")
            if filters:
                print(f"Active filters: {', '.join(filters)}")
            print_summary(ordered, top_n=args.top)
            return 0

        total_count, filtered_count, ordered = render_from_cache(
            paths.cache_json,
            ranked_csv,
            sort_mode=args.sort,
            filters=filters,
        )
        print(f"Cache: {paths.cache_json}")
        print(f"CSV:   {ranked_csv}")
        print(f"Cached PRs: {total_count}")
        print(f"Rows written: {filtered_count}")
        if filters:
            print(f"Active filters: {', '.join(filters)}")
        print_summary(ordered, top_n=args.top)
        return 0
    except GhCommandError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
