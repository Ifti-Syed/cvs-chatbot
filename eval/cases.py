"""
Evaluation check types and TestCase definition.

All checks operate on the final answer string produced by the chatbot.
Each check returns a CheckResult (passed/failed + human-readable message).

Check types
───────────
  MustContain(value)           answer must contain the substring (case-insensitive by default)
  MustNotContain(value)        answer must NOT contain the substring
  MustContainAny(values)       answer must contain at least one of the given substrings
  MustCiteSource()             answer must end with a citation in (Source: ...) format
  MustNotBeVague(exact, vague) if the answer contains `vague`, it must also contain `exact`
                               (catches "as required by the standard" without the actual value)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# CheckResult
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    check_name: str
    passed: bool
    message: str

    def __str__(self) -> str:
        symbol = "✓" if self.passed else "✗"
        return f"  {symbol} [{self.check_name}] {self.message}"


# ---------------------------------------------------------------------------
# Check types
# ---------------------------------------------------------------------------

@dataclass
class MustContain:
    """Answer must contain `value` as a substring."""
    value: str
    case_sensitive: bool = False

    def check(self, answer: str) -> CheckResult:
        haystack = answer if self.case_sensitive else answer.lower()
        needle   = self.value if self.case_sensitive else self.value.lower()
        passed   = needle in haystack
        return CheckResult(
            check_name=f"MustContain({self.value!r})",
            passed=passed,
            message=(
                f"Found {self.value!r} in answer."
                if passed
                else f"MISSING {self.value!r} — answer did not contain this value."
            ),
        )


@dataclass
class MustNotContain:
    """Answer must NOT contain `value` as a substring."""
    value: str
    case_sensitive: bool = False

    def check(self, answer: str) -> CheckResult:
        haystack = answer if self.case_sensitive else answer.lower()
        needle   = self.value if self.case_sensitive else self.value.lower()
        found    = needle in haystack
        return CheckResult(
            check_name=f"MustNotContain({self.value!r})",
            passed=not found,
            message=(
                f"Correctly absent: {self.value!r}."
                if not found
                else f"FOUND forbidden phrase {self.value!r} in answer."
            ),
        )


@dataclass
class MustContainAny:
    """Answer must contain at least one of the given substrings."""
    values: list[str]
    case_sensitive: bool = False

    def check(self, answer: str) -> CheckResult:
        haystack = answer if self.case_sensitive else answer.lower()
        for v in self.values:
            needle = v if self.case_sensitive else v.lower()
            if needle in haystack:
                return CheckResult(
                    check_name=f"MustContainAny({self.values!r})",
                    passed=True,
                    message=f"Found {v!r} (one of the accepted values).",
                )
        return CheckResult(
            check_name=f"MustContainAny({self.values!r})",
            passed=False,
            message=f"None of {self.values!r} found in answer.",
        )


@dataclass
class MustCiteSource:
    """
    Answer must end with a citation in the format:
        (Source: Document Name, Page X)
        (Sources: Doc A, Page X; Doc B, Page Y)
    """
    def check(self, answer: str) -> CheckResult:
        # Allow optional whitespace/newlines after the closing paren
        pattern = r"\(Source[s]?:.+?\)\s*$"
        passed  = bool(re.search(pattern, answer.strip(), re.IGNORECASE | re.DOTALL))
        return CheckResult(
            check_name="MustCiteSource",
            passed=passed,
            message=(
                "Citation found at end of answer."
                if passed
                else "MISSING citation — expected (Source: ..., Page X) at end of answer."
            ),
        )


@dataclass
class MustNotBeVague:
    """
    If the answer contains `vague_phrase`, it must also contain `exact_value`.

    Use this to catch cases where the model replaced a specific number with
    a vague regulatory reference, e.g.:
        exact_value  = "75%"
        vague_phrase = "as required by the standard"
    If the answer says "as required by the standard" it MUST also say "75%".
    """
    exact_value: str
    vague_phrase: str
    case_sensitive: bool = False

    def check(self, answer: str) -> CheckResult:
        name = f"MustNotBeVague(exact={self.exact_value!r}, vague={self.vague_phrase!r})"
        haystack = answer if self.case_sensitive else answer.lower()
        vague    = self.vague_phrase if self.case_sensitive else self.vague_phrase.lower()
        exact    = self.exact_value  if self.case_sensitive else self.exact_value.lower()

        vague_present = vague in haystack
        exact_present = exact in haystack

        if not vague_present:
            # Vague phrase not used — nothing to complain about
            return CheckResult(check_name=name, passed=True,
                               message=f"Vague phrase {self.vague_phrase!r} not used.")
        if exact_present:
            # Vague phrase used but exact value also present — acceptable
            return CheckResult(check_name=name, passed=True,
                               message=f"Vague phrase used but exact value {self.exact_value!r} also present.")
        # Vague phrase used WITHOUT the exact value — fail
        return CheckResult(
            check_name=name,
            passed=False,
            message=(
                f"Answer contains vague phrase {self.vague_phrase!r} "
                f"but is MISSING the exact value {self.exact_value!r}."
            ),
        )


# ---------------------------------------------------------------------------
# TestCase
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    """A single evaluation case: one question with a list of answer checks."""
    question: str
    checks: list[Any]                   # list of check objects defined above
    description: str = ""               # short label shown in the report
    tags: list[str] = field(default_factory=list)   # e.g. ["smoke", "exact-value"]

    def run_checks(self, answer: str) -> list[CheckResult]:
        return [c.check(answer) for c in self.checks]
