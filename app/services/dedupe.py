"""
Global dedupe over the combined, extracted feature list (requirements.md §2.6).

Rewritten to be SCALABLE and LangSmith-traced:
- Built on LangChain `ChatAnthropic` + `.with_structured_output(DedupeDecision)`
  → schema-guaranteed output (no ```json parsing) AND it now shows up in LangSmith.
- The model returns DECISIONS ONLY — `keep_ids` (uniques, by index), `merged`
  (only the duplicate groups it collapses), and `drop_ids` (redundant coarse
  summaries). Python assembles the final list. Output stays tiny → never
  truncates, at any document size (the old version echoed the whole list and
  silently truncated on large BRDs).
- Merge-only: the "split bundled entries" behaviour is gone — the extraction
  verifier owns all granularity/over-split decisions now.

Public signature is unchanged: `dedupe_features(list[dict]) -> list[dict]`.
"""

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

from app.config import settings
from app.models.extraction_schemas import DedupeDecision


_llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    temperature=0,
    api_key=settings.ANTHROPIC_API_KEY,
)
_dedupe_llm = _llm.with_structured_output(DedupeDecision)


DEDUPE_SYSTEM_PROMPT = """You are given a NUMBERED list of software features extracted from different sections of the SAME project document. Because sections overlap, and because summary/scope sections restate detailed requirements, the list contains DUPLICATES: the same capability appearing more than once.

Your ONLY job is to identify duplicates. This is a duplicate-removal task — NOT summarization, and NOT splitting. Do not invent, rename, or re-scope features beyond what merging requires.

Return DECISIONS ONLY (never re-list every feature). For each feature index you decide one of three things:

1. keep_ids — the feature is UNIQUE (nothing else in the list is the same capability). Keep it unchanged.

2. merged — the feature is a TRUE DUPLICATE of one or more others: the SAME single capability, just re-extracted in slightly different words. Put their indices together in one `merged` group and synthesize ONE feature: the single most complete plain-language description (lose NO information), the most appropriate domain, and combined tech_details (comma-separated, no duplication).

3. drop_ids — the feature is a REDUNDANT COARSE/SUMMARY item that is already FULLY covered by SEVERAL finer features present in the list. Example: a summary bullet "Cart, Wishlist & Checkout" when the list also has separate "Add to Cart", "Cart Management", "Wishlist", and "Checkout" features. Drop the coarse one (it adds nothing) and keep the finer ones. Only drop when NO information is lost.

CRITICAL invariant: EVERY index in the list must appear EXACTLY ONCE — either in keep_ids, or inside exactly one merged group's from_ids, or in drop_ids. Never in two of them, never in none.

When unsure whether two features are the same capability or two different-but-related capabilities, KEEP THEM SEPARATE (put both in keep_ids). It is far better to leave a near-duplicate than to wrongly collapse two distinct capabilities — a later step matches these against what new clients ask for, and clients ask for capabilities individually."""


def _assemble(features: list[dict], decision: DedupeDecision) -> list[dict]:
    """Build the final feature list from the model's decisions, defensively.

    Guarantees no feature is silently lost: any index the model failed to account
    for is kept as-is (better an accidental duplicate than a dropped capability).
    """
    n = len(features)
    result: list[dict] = []
    used: set[int] = set()

    # 1. Merged groups → the single synthesized feature.
    for m in decision.merged:
        ids = [i for i in m.from_ids if 0 <= i < n and i not in used]
        if not ids:
            continue
        used.update(ids)
        if len(ids) == 1:
            # Not actually a group — keep the original rather than a rewrite.
            result.append(features[ids[0]])
        else:
            result.append({
                "name": m.name,
                "description": m.description,
                "domain": m.domain,
                "tech_details": m.tech_details,
            })

    # 2. Explicit drops → removed (just mark them used).
    for i in decision.drop_ids:
        if 0 <= i < n:
            used.add(i)

    # 3. keep_ids → original features, unchanged.
    for i in decision.keep_ids:
        if 0 <= i < n and i not in used:
            used.add(i)
            result.append(features[i])

    # 4. Safety net: anything the model forgot → keep it (never lose a feature).
    for i in range(n):
        if i not in used:
            result.append(features[i])

    return result


async def dedupe_features(features: list[dict]) -> list[dict]:
    if len(features) <= 1:
        return features

    # Present the list with a stable index id per feature; the model references
    # these ids in keep_ids / from_ids / drop_ids.
    listing = "\n".join(
        f"[{i}] name: {f.get('name', '')} | domain: {f.get('domain', '')} | "
        f"description: {f.get('description', '')}"
        for i, f in enumerate(features)
    )

    try:
        decision: DedupeDecision = await _dedupe_llm.ainvoke(
            [
                SystemMessage(DEDUPE_SYSTEM_PROMPT),
                HumanMessage(f"Feature list ({len(features)} items):\n{listing}"),
            ]
        )
    except Exception:
        # On any failure, fall back to the raw list rather than losing everything.
        return features

    return _assemble(features, decision)
