"""
Pydantic schemas for the agentic feature-extraction pipeline.

These models are the *structured-output contracts* for the extraction and
verification LLM calls. With Pydantic + structured outputs the model is forced to
return schema-valid JSON, so we no longer parse raw text with
`.replace("```json", "")` + `json.loads` (which fails silently on bad output).

NOTE: the Field descriptions are not just docs — with structured outputs the model
reads them to decide how to fill each field. Treat them as part of the prompt.

Mirrors requirements.md §2 (extraction) and §2.4 (verifier output).
"""

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Extraction output                                                           #
# --------------------------------------------------------------------------- #

class Feature(BaseModel):
    """One distinct capability extracted from a project document."""

    name: str = Field(
        description="Short human label for the feature, 3-6 words.",
    )
    description: str = Field(
        description=(
            "Self-contained, plain-language sentence describing what the feature "
            "does for its users and WHY it matters. No technical jargon. Include the "
            "domain context (e.g. 'delivery vehicle', 'student', 'invoice') so the "
            "meaning is unambiguous on its own. This is the most important field."
        ),
    )
    domain: str = Field(
        description=(
            "The business area this feature belongs to "
            "(e.g. 'logistics', 'authentication', 'payments', 'education')."
        ),
    )
    tech_details: str = Field(
        default="",
        description=(
            "Any specific technical implementation details mentioned "
            "(e.g. 'JWT with refresh tokens', 'Razorpay integration'). "
            "Empty string if none. Internal reference only."
        ),
    )


class FeatureList(BaseModel):
    """
    Wrapper around a list of features.

    Structured outputs need a top-level object (not a bare array), so the
    extraction call returns this and we read `.features`.
    """

    features: list[Feature] = Field(
        default_factory=list,
        description="All distinct features found in the document section.",
    )


# --------------------------------------------------------------------------- #
# Verifier output (requirements.md §2.4)                                       #
# --------------------------------------------------------------------------- #

class MissedFeature(BaseModel):
    """A capability present in the text that extraction failed to capture."""

    hint: str = Field(
        description=(
            "Short description of a capability that IS present in the chunk text "
            "but was not extracted as a feature."
        ),
    )


class WronglySplit(BaseModel):
    """A single capability that was incorrectly broken into several features."""

    feature_names: list[str] = Field(
        description="Names of the extracted features that are really ONE capability.",
    )
    reason: str = Field(
        description="Why these belong together as a single feature.",
    )


class VerifyVerdict(BaseModel):
    """
    The verifier's judgment on one chunk's extraction.

    `clean == True` with empty lists means the extraction is good and the loop
    can stop. Otherwise `missed` / `wrongly_split` drive one re-extraction round.
    """

    clean: bool = Field(
        description=(
            "True ONLY if nothing was missed AND nothing was wrongly split. "
            "If True, `missed` and `wrongly_split` must both be empty."
        ),
    )
    missed: list[MissedFeature] = Field(
        default_factory=list,
        description="Capabilities present in the text but not extracted.",
    )
    wrongly_split: list[WronglySplit] = Field(
        default_factory=list,
        description="Groups of features that should be merged into one.",
    )


# --------------------------------------------------------------------------- #
# Dedupe output (requirements.md §2.6)                                         #
# --------------------------------------------------------------------------- #
# Dedupe runs AFTER extraction, over the combined feature list. Each input
# feature is given a temporary integer id = its index in the list. The model
# returns only DECISIONS (not the whole list): which ids are unique, and which
# groups it merged. Python assembles the final list — so the output stays tiny
# and never truncates, at any document size.

class MergedFeature(BaseModel):
    """One merged feature, synthesized from a group of true duplicates."""

    from_ids: list[int] = Field(
        description=(
            "The ids (indices) of the input features that are the SAME single "
            "capability and are being merged into this one feature. Must contain "
            "at least two ids."
        ),
    )
    name: str = Field(description="Name for the merged feature (pick the clearest).")
    description: str = Field(
        description=(
            "The single most complete, self-contained, plain-language description "
            "for the merged capability — a reader must lose NO information versus "
            "the originals."
        ),
    )
    domain: str = Field(description="The single most appropriate domain for the group.")
    tech_details: str = Field(
        default="",
        description="Combined tech_details from the group, comma-separated, no duplication.",
    )


class DedupeDecision(BaseModel):
    """The model's dedupe decision — DECISIONS ONLY, never the whole list."""

    keep_ids: list[int] = Field(
        default_factory=list,
        description=(
            "Ids (indices) of features that are UNIQUE — no duplicate of them exists "
            "in the list. Kept unchanged."
        ),
    )
    merged: list[MergedFeature] = Field(
        default_factory=list,
        description=(
            "Groups of TRUE duplicates (the same single capability appearing more "
            "than once, in slightly different words) merged into one synthesized "
            "feature. Empty if there are no duplicate groups."
        ),
    )
    drop_ids: list[int] = Field(
        default_factory=list,
        description=(
            "Ids of REDUNDANT coarse/summary features that are FULLY covered by "
            "several finer features already present in the list (e.g. a 'Cart, "
            "Wishlist & Checkout' summary when separate 'Add to Cart', 'Wishlist', "
            "and 'Checkout' features exist). These are removed entirely — NOT merged. "
            "Only use when NO information is lost. Empty if none."
        ),
    )
    # INVARIANT: every input id appears EXACTLY ONCE across keep_ids, the from_ids
    # of the merged groups, and drop_ids — never in two of them, never in none.
