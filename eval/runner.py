"""
RAG Evaluation Runner

Runs every TestCase in eval/test_cases.py against the live chatbot pipeline.

For each question it logs:
  - Query expansions generated for that question
  - Retrieved chunks before LLM relevance selection (pre-rerank)
  - Final chunks after LLM relevance selection (post-rerank)
  - The full answer text
  - Pass / Fail status of every check

Usage
-----
  # Run all tests
  python -m eval.runner

  # Run only tests tagged "smoke"
  python -m eval.runner --tags smoke

  # Run all tests and save a JSON report
  python -m eval.runner --report eval/reports/run_001.json

  # Show full answer text (default: truncated to 400 chars)
  python -m eval.runner --full-answer

  # Suppress chunk details (show only answer + checks per question)
  python -m eval.runner --quiet
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path when run as `python -m eval.runner`
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import get_settings
from backend.services.chat_service import ChatService
from backend.services.openai_service import OpenAIService
from backend.services.vector_db import QdrantService
from eval.cases import CheckResult, TestCase
from eval.test_cases import ALL_TESTS

# ---------------------------------------------------------------------------
# ANSI colours (degrade gracefully on terminals that don't support them)
# ---------------------------------------------------------------------------

def _supports_ansi() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if sys.platform == "win32":
        # Enable VT100 on Windows 10+
        try:
            import ctypes
            kernel = ctypes.windll.kernel32
            kernel.SetConsoleMode(kernel.GetStdHandle(-11), 7)
            return True
        except Exception:
            return False
    return sys.stdout.isatty()

_USE_COLOR = _supports_ansi()

def _ansi(code: str) -> str:
    return f"\033[{code}m" if _USE_COLOR else ""

RESET  = _ansi("0")
BOLD   = _ansi("1")
RED    = _ansi("91")
GREEN  = _ansi("92")
CYAN   = _ansi("96")
DIM    = _ansi("2")

def _pass(t: str) -> str: return f"{GREEN}{t}{RESET}"
def _fail(t: str) -> str: return f"{RED}{t}{RESET}"
def _head(t: str) -> str: return f"{BOLD}{CYAN}{t}{RESET}"
def _dim(t: str)  -> str: return f"{DIM}{t}{RESET}"


# ---------------------------------------------------------------------------
# Chunk printer
# ---------------------------------------------------------------------------

def _print_chunks(chunks: list[dict], label: str, quiet: bool) -> None:
    if quiet or not chunks:
        return
    print(f"  {_dim(label)} ({len(chunks)} chunks)")
    for i, c in enumerate(chunks, 1):
        doc     = c.get("document_name", "?")
        page    = c.get("page_number", "?")
        rrf     = c.get("rrf_score")
        score   = f"rrf={rrf:.4f}" if rrf is not None else f"score={c.get('score', 0):.4f}"
        rtype   = c.get("retrieval_type", "dense")
        preview = c.get("text_preview", "")[:120].replace("\n", " ")
        print(f"    [{i}] {doc} p{page} | {score} | {rtype}")
        print(f"        {_dim(preview + '...')}")


# ---------------------------------------------------------------------------
# Single test execution
# ---------------------------------------------------------------------------

async def run_one(
    case: TestCase,
    service: ChatService,
    full_answer: bool,
    quiet: bool,
) -> dict[str, Any]:
    """
    Run a single TestCase.
    Returns a result dict suitable for JSON serialisation.
    """
    print()
    desc  = case.description or case.question[:80]
    ruler = "-" * max(0, 70 - len(desc))
    print(_head(f"-- {desc} {ruler}"))
    if case.tags:
        print(_dim(f"  tags: {', '.join(case.tags)}"))
    print(f"  {BOLD}Q:{RESET} {case.question}")

    # Run the full pipeline with debug trace
    try:
        result = await service.get_response_debug(case.question)
    except Exception as exc:
        print(_fail(f"  ERROR: pipeline raised {type(exc).__name__}: {exc}"))
        return {
            "description": desc,
            "question": case.question,
            "error": str(exc),
            "checks": [],
            "passed": False,
        }

    answer      = result["answer"]
    expansions  = result["query_expansions"]
    pre_rerank  = result["pre_rerank"]
    post_rerank = result["post_rerank"]

    # Query expansions
    if not quiet and expansions:
        print(f"  {_dim('Expanded queries:')}")
        for i, eq in enumerate(expansions, 1):
            print(f"    {_dim(str(i) + '.')} {_dim(eq)}")

    # Chunk details
    _print_chunks(pre_rerank,  "Pre-rerank candidates:",        quiet)
    _print_chunks(post_rerank, "Post-rerank (sent to LLM):",    quiet)

    # Answer
    print(f"\n  {BOLD}Answer:{RESET}")
    displayed = answer if full_answer else (
        answer[:400] + ("..." if len(answer) > 400 else "")
    )
    for line in textwrap.wrap(
        displayed, width=90,
        initial_indent="    ", subsequent_indent="    ",
    ):
        print(line)

    # Checks
    print(f"\n  {BOLD}Checks:{RESET}")
    results: list[CheckResult] = case.run_checks(answer)
    all_passed = True
    for r in results:
        if r.passed:
            print(_pass(f"  PASS [{r.check_name}] {r.message}"))
        else:
            print(_fail(f"  FAIL [{r.check_name}] {r.message}"))
            all_passed = False

    status = _pass("PASS") if all_passed else _fail("FAIL")
    print(f"\n  Status: {status}")

    return {
        "description": desc,
        "question": case.question,
        "tags": case.tags,
        "query_expansions": expansions,
        "pre_rerank_count": len(pre_rerank),
        "post_rerank_count": len(post_rerank),
        "pre_rerank": pre_rerank,
        "post_rerank": post_rerank,
        "answer": answer,
        "checks": [
            {"name": r.check_name, "passed": r.passed, "message": r.message}
            for r in results
        ],
        "passed": all_passed,
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

async def main(args: argparse.Namespace) -> None:
    tag_filter: list[str] = (
        [t.strip() for t in args.tags.split(",") if t.strip()]
        if args.tags else []
    )

    # Filter test cases by tag
    cases: list[TestCase] = ALL_TESTS
    if tag_filter:
        cases = [c for c in cases if any(t in c.tags for t in tag_filter)]
        if not cases:
            print(_fail(f"No test cases matched tags: {tag_filter}"))
            sys.exit(1)
        print(_dim(f"Running {len(cases)} test(s) matching tags: {tag_filter}"))

    # Initialise services (no collection init needed — collection already exists)
    config   = get_settings()
    qdrant   = QdrantService(config)
    openai_s = OpenAIService(config)
    service  = ChatService(qdrant, openai_s, config)

    sep = "=" * 72
    print(_head(f"\n{sep}"))
    print(_head(
        f"  RAG Evaluation Runner  |  "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    ))
    print(_head(
        f"  Chat model : {config.openai_chat_model}  "
        f"Rerank model: {config.openai_rerank_model}"
    ))
    print(_head(
        f"  top_k={config.top_k_results}  "
        f"rerank_top_k={config.rerank_top_k}  "
        f"hybrid={config.enable_hybrid_search}  "
        f"reranking={config.enable_reranking}  "
        f"threshold={config.min_score_threshold}"
    ))
    print(_head(f"{sep}\n"))

    all_results: list[dict] = []
    for case in cases:
        result = await run_one(
            case,
            service,
            full_answer=args.full_answer,
            quiet=args.quiet,
        )
        all_results.append(result)

    # Summary
    total  = len(all_results)
    passed = sum(1 for r in all_results if r.get("passed"))
    failed = total - passed

    print()
    print(_head(sep))
    print(_head("  SUMMARY"))
    print(_head(sep))
    print(f"  Total  : {total}")
    print((_pass if passed else str)(f"  Passed : {passed}"))
    print((_fail if failed else str)(f"  Failed : {failed}"))
    print()

    for r in all_results:
        status = _pass("PASS") if r.get("passed") else _fail("FAIL")
        print(f"  {status}  {r['description']}")
        if not r.get("passed"):
            for c in r.get("checks", []):
                if not c["passed"]:
                    print(_fail(f"       FAIL {c['name']}: {c['message']}"))

    print()

    # JSON report
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "run_at": datetime.now(timezone.utc).isoformat(),
            "config": {
                "openai_chat_model":    config.openai_chat_model,
                "openai_rerank_model":  config.openai_rerank_model,
                "top_k_results":        config.top_k_results,
                "rerank_top_k":         config.rerank_top_k,
                "enable_hybrid_search": config.enable_hybrid_search,
                "enable_reranking":     config.enable_reranking,
                "min_score_threshold":  config.min_score_threshold,
                "chunk_size":           config.chunk_size,
            },
            "summary": {"total": total, "passed": passed, "failed": failed},
            "results": all_results,
        }
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  Report saved: {report_path}")

    sys.exit(0 if failed == 0 else 1)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RAG pipeline evaluation runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tags",
        default="",
        metavar="TAG[,TAG]",
        help=(
            "Comma-separated tags to filter which test cases run. "
            "Example: --tags smoke,exact-value"
        ),
    )
    parser.add_argument(
        "--report",
        default="",
        metavar="PATH",
        help=(
            "Write a JSON report to this file. "
            "Example: --report eval/reports/run_001.json"
        ),
    )
    parser.add_argument(
        "--full-answer",
        action="store_true",
        help="Print the full answer text instead of truncating to 400 chars.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Hide chunk details and query expansions; show only answer + checks.",
    )

    args = parser.parse_args()
    asyncio.run(main(args))
