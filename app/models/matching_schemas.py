"""
Structured-output schemas for the composite matching graph (requirements.md §3).

Each LLM node in the matching graph returns one of these (via
`.with_structured_output(...)`), so outputs are schema-guaranteed — no fragile
```json parsing. The graph STATE itself (MatchState) lives in matching_graph.py.

Flow the schemas map to:
  search -> coverage(CoverageDecision) -> classify(ClassifyVerdict)
        -> exact_match(MatchExplanation)  OR  modify(ModificationOutput) ⇄ critic(CriticVerdict)
"""

from typing import Literal
from pydantic import BaseModel, Field


class CoverageDecision(BaseModel):
    """`coverage` node: which candidate feature(s) cover the requirement?"""

    relevant: bool = Field(
        description=(
            "True if at least one retrieved candidate is REUSABLE for the requirement "
            "— either as-is or with modification. False if none are genuinely relevant."
        ),
    )
    selected_ids: list[int] = Field(
        default_factory=list,
        description=(
            "Indices (from the numbered candidate list) of the features that TOGETHER "
            "cover the requirement. One index if a single feature fully satisfies it; "
            "several if it genuinely takes a combination. Empty if not relevant."
        ),
    )
    confidence: float = Field(
        description="0.0-1.0 confidence that the selected set covers the requirement (one overall number).",
    )
    reasoning: str = Field(description="Brief explanation of the selection.")


class ClassifyVerdict(BaseModel):
    """`classify` node: is the selected set an exact match or does it need work?"""

    status: Literal["exact_match", "needs_modification"] = Field(
        description=(
            "'exact_match' if the selected feature(s) already do what the requirement "
            "needs, reusable essentially as-is; 'needs_modification' if the underlying "
            "logic is reusable but must be adapted (e.g. different domain, extra behaviour)."
        ),
    )
    confidence: float = Field(description="0.0-1.0 confidence in this classification.")


class MatchExplanation(BaseModel):
    """`exact_match` node: a short client-facing 'why this matches'."""

    explanation: str = Field(
        description=(
            "A short, plain-language explanation for the client of why the existing "
            "feature(s) already satisfy this requirement."
        ),
    )


class ModificationOutput(BaseModel):
    """`modify` node: what needs to change to adapt the existing feature(s)."""

    modification: str = Field(
        description=(
            "Plain-language, concrete description of WHAT must change to adapt the "
            "existing feature(s) to the requirement (1-3 sentences)."
        ),
    )


class CriticVerdict(BaseModel):
    """`critic` node: is the modification description actually useful?"""

    valid: bool = Field(
        description=(
            "True if the modification description is concrete and useful. False if it "
            "is vacuous, empty, contradictory, or merely says the feature is not needed."
        ),
    )
    reason: str = Field(description="Why it is valid or not.")
