"""
Evaluation test cases.

This suite is designed to validate a general-purpose document QA system.
It includes:
- exact value extraction
- hallucination prevention
- reasoning quality
- citation correctness
- consistency across runs

These tests are NOT domain-specific and should work across
technical documents, manuals, reports, policies, and datasheets.
"""

from eval.cases import (
    MustContain,
    MustContainAny,
    MustCiteSource,
    MustNotBeVague,
    MustNotContain,
    TestCase,
)

# ---------------------------------------------------------------------------
# SMOKE TESTS — basic sanity checks
# ---------------------------------------------------------------------------

SMOKE_TESTS: list[TestCase] = [
    TestCase(
        description="Answer must include citation and avoid generic phrasing",
        question="What is this document about?",
        tags=["smoke", "generic"],
        checks=[
            MustCiteSource(),
            MustNotContain("According to the document"),
            MustNotContain("Based on the provided content"),
        ],
    ),
]

# ---------------------------------------------------------------------------
# EXACT VALUE EXTRACTION — core intelligence tests
# ---------------------------------------------------------------------------

EXACT_VALUE_TESTS: list[TestCase] = [
    TestCase(
        description="Cross-section area — must state 75%, not vague",
        question="What is the cross section area maitain during the test?",
        tags=["exact-value", "cvs"],
        checks=[
            MustContain("75%"),
            MustCiteSource(),
            MustNotBeVague(
                exact_value="75%",
                vague_phrase="as required by the standard",
            ),
            MustNotBeVague(
                exact_value="75%",
                vague_phrase="maintained as required",
            ),
        ],
    ),
    TestCase(
        description="Reworded cross-section question (robust extraction)",
        question="During fire testing for smoke extract performance, what minimum proportion of the duct’s cross-sectional area must remain effective?",
        tags=["exact-value", "cvs"],
        checks=[
            MustContain("75%"),
            MustCiteSource(),
        ],
    ),
    TestCase(
        description="Fire rating duration must include numeric value",
        question="What fire resistance or smoke extract duration is specified?",
        tags=["exact-value", "generic"],
        checks=[
            MustContainAny([
                "hour", "hours", "min", "minutes",
                "60", "120", "180", "240"
            ]),
            MustCiteSource(),
            MustNotContain("long duration"),
            MustNotContain("extended period"),
        ],
    ),
    TestCase(
        description="Generic numeric extraction",
        question="What numeric values are specified in the document?",
        tags=["exact-value", "generic"],
        checks=[
            MustContainAny(["%", "mm", "hour", "Pa", "kPa"]),
            MustCiteSource(),
        ],
    ),
    TestCase(
        description="Temperature value (or correctly missing)",
        question="What temperature rating or limit is specified?",
        tags=["exact-value", "generic"],
        checks=[
            MustContainAny([
                "°C", "°F", "degrees",
                "not specified", "not mentioned", "no information"
            ]),
            MustCiteSource(),
        ],
    ),
]

# ---------------------------------------------------------------------------
# REASONING TESTS — explanation quality
# ---------------------------------------------------------------------------

REASONING_TESTS: list[TestCase] = [
    TestCase(
        description="Why question — should explain or state limitation",
        question="Why is it critical for the duct system to maintain its internal cross-sectional area during a fire scenario?",
        tags=["reasoning"],
        checks=[
            MustContainAny([
                "extraction",
                "performance",
                "smoke",
                "maintain",
                "function"
            ]),
            MustCiteSource(),
        ],
    ),
]

# ---------------------------------------------------------------------------
# HALLUCINATION TESTS — prevent fabricated answers
# ---------------------------------------------------------------------------

HALLUCINATION_TESTS: list[TestCase] = [
    TestCase(
        description="Out-of-scope pricing — must not fabricate",
        question="What is the price of the product in US dollars?",
        tags=["hallucination"],
        checks=[
            MustNotContain("$"),
            MustNotContain("USD"),
            MustContainAny([
                "not covered",
                "not available",
                "not mentioned",
                "not specified",
                "no information",
                "does not provide",
                "does not specify",
            ]),
        ],
    ),
    TestCase(
        description="Installation requirements — no guessing",
        question="What are the full installation requirements?",
        tags=["hallucination"],
        checks=[
            MustCiteSource(),
            MustNotContain("typically"),
            MustNotContain("generally"),
            MustNotContain("usually"),
        ],
    ),
]

# ---------------------------------------------------------------------------
# CONSISTENCY TESTS — stability across runs
# ---------------------------------------------------------------------------

CONSISTENCY_TESTS: list[TestCase] = [
    TestCase(
        description="Consistency — cross-section must always return 75%",
        question="What is the cross-sectional area requirement during testing?",
        tags=["consistency"],
        checks=[
            MustContain("75%"),
            MustCiteSource(),
        ],
    ),
]

# ---------------------------------------------------------------------------
# CITATION QUALITY
# ---------------------------------------------------------------------------

CITATION_TESTS: list[TestCase] = [
    TestCase(
        description="Citation must be specific and not generic wording",
        question="Which standard or certification applies to the system?",
        tags=["citation"],
        checks=[
            MustCiteSource(),
            MustNotContain("the document states"),
            MustNotContain("according to the document"),
        ],
    ),
]

# ---------------------------------------------------------------------------
# ALL TESTS
# ---------------------------------------------------------------------------

ALL_TESTS: list[TestCase] = (
    SMOKE_TESTS
    + EXACT_VALUE_TESTS
    + REASONING_TESTS
    + HALLUCINATION_TESTS
    + CONSISTENCY_TESTS
    + CITATION_TESTS
)