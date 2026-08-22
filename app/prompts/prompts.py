RELEVANCE_SYSTEM_PROMPT = """You are matching a NEW client's requirement against a library of features a software agency has already built.
 
You will be given:
1. The client's requirement (what they want).
2. A list of candidate features the agency has built before (retrieved by semantic similarity).
 
Your job: decide whether ANY single candidate is genuinely the SAME capability the client is asking for — not merely related or in the same general area.
 
Be strict about two things:
- DOMAIN: a candidate can use similar words but belong to a different domain (e.g. "estimated arrival time for a delivery vehicle" vs "estimated completion time for a course"). Similar words but different domain = NOT a match.
- SAME CAPABILITY: the candidate must do essentially what the client wants, not just something adjacent.
 
If one candidate is a genuine match, pick the SINGLE best one.
If none genuinely match, say so.
 
Return ONLY this JSON, nothing else:
{"relevant": true, "matched_index": <0-based index of the best candidate>, "confidence": <0.0-1.0>}
OR
{"relevant": false}
"""
 

NODE1_RELEVANCE_SYSTEM_PROMPT = """You are matching a NEW client's requirement against a library of features a software agency has already built.

You will be given:
1. The client's requirement (what they want).
2. A list of candidate features the agency has built before (retrieved by semantic similarity).

Your job: decide whether ANY single candidate is REUSABLE for the client's requirement — meaning the underlying logic or capability is similar enough that the agency could reuse that past work, either as-is or with modification.

How to judge (this is important):
- Focus on the underlying LOGIC / CAPABILITY, not surface wording and not the business domain.
- A candidate from a DIFFERENT domain can still be a match if its core logic is reusable. Example: a "customers earn loyalty points redeemable for discounts" feature IS reusable for a "students earn points redeemable for rewards" requirement — same points-accrual-and-redemption logic, just a different domain. This should be considered relevant (it will be marked as needing modification later).
- Only reject a candidate when its underlying logic is genuinely DIFFERENT, not merely because the domain differs. Example: "estimating a delivery vehicle's arrival using GPS and route data" is NOT reusable for "estimating when a student will finish a course", because the actual logic (physical location tracking vs. lesson-progress tracking) is fundamentally different — even though both are called an "estimate".

So: similar logic (even across domains) = relevant. Different logic = not relevant. Do NOT reject on domain difference alone.

If one or more candidates are reusable, pick the SINGLE best one (the closest in underlying logic).
If none are genuinely reusable, say so.

Return ONLY this JSON, nothing else:
{"relevant": true, "matched_index": <0-based index of the best candidate>, "confidence": <0.0-1.0>}
OR
{"relevant": false}
"""





REPHRASE_SYSTEM_PROMPT = """A search for a reusable software feature returned no relevant match for a client's requirement. The search is semantic, based on the wording of the requirement.

Rewrite the requirement below using DIFFERENT words and phrasing, while keeping the exact same meaning and intent. Focus on the underlying capability or logic, described in plain language. The goal is to phrase it differently enough that a fresh semantic search might surface features that the original wording missed.

Return ONLY the rewritten requirement as a single plain sentence. No preamble, no quotes, no explanation."""





CLASSIFY_SYSTEM_PROMPT = """A client's requirement has been matched to a feature the agency already built. Both are given below.

Decide how reusable the existing feature is for this requirement:

- "exact_match": the existing feature already does what the client wants. It could be reused essentially as-is, with no meaningful changes to its logic or behaviour. Minor cosmetic differences (wording, labels) still count as exact.

- "needs_modification": the underlying logic is reusable, but the feature would have to be adapted to fit the client's requirement. This is common when the DOMAIN differs (e.g. reusing a restaurant loyalty-points system for a student rewards system) or when the client needs extra behaviour the existing feature doesn't have.

Judge based on the underlying logic and how much adaptation is realistically needed.

Return ONLY this JSON, nothing else:
{"status": "exact_match" | "needs_modification", "confidence": <0.0-1.0>}"""


MODIFICATION_SYSTEM_PROMPT = """A client's requirement can be met by adapting a feature the agency already built, but it needs some modification. Both are given below.

Explain clearly and concisely WHAT would need to change to adapt the existing feature to the client's requirement. Focus on the practical differences a developer or the client would care about (e.g. different domain concepts, extra behaviour needed, data differences). Keep it to 1-3 short sentences, in plain language a non-technical person can follow.

Return ONLY the explanation text. No preamble, no JSON, no quotes."""


# =========================================================================== #
# Composite matching graph (requirements.md §3)                               #
# These power the structured-output nodes, so they carry NO "return JSON"      #
# instruction — the Pydantic schema enforces the shape. They also reason over  #
# a SELECTED SET of features (single OR combined), not just one candidate.     #
# =========================================================================== #

COVERAGE_SYSTEM_PROMPT = """You are matching a NEW client's requirement against a library of features a software agency has already built. You are given the requirement and a NUMBERED list of candidate features (retrieved by semantic similarity).

Decide which candidate feature(s), if any, are REUSABLE to satisfy the requirement:
- Focus on the underlying LOGIC / capability, NOT surface wording and NOT the business domain. A candidate from a DIFFERENT domain can still be reusable if its core logic fits (it will be marked as needing modification later). Only reject a candidate when its underlying logic is genuinely different — not merely because the domain differs.
- If ONE candidate fully covers the requirement, select just that one.
- If no single candidate covers it but a COMBINATION does, select the MINIMAL set of candidates that TOGETHER cover it.
- If none are genuinely reusable, mark it not relevant and leave the selection empty.

Give one overall confidence (0.0-1.0) that the selected set covers the requirement."""


CLASSIFY_SET_SYSTEM_PROMPT = """A client's requirement has been matched to one or more features the agency already built (given below as a SET). Decide how reusable that set is for this requirement:

- "exact_match": the selected feature(s) already do what the client wants and could be reused essentially as-is, with no meaningful change to their logic or behaviour. Cosmetic differences — wording, labels, or a different business domain that does NOT change how the feature actually works — still count as exact. A domain difference on its own does not make it a modification if the feature would behave identically.
- "needs_modification": the underlying logic is reusable, but the feature(s) must actually be adapted to fit the requirement — either because the requirement needs extra behaviour the feature(s) lack, or because the domain change forces real changes to the feature's logic, data, or rules (e.g. reusing a restaurant loyalty-points system for a student rewards system, where the entities and rules genuinely differ). A domain difference alone, with no change to how the feature works, is NOT a modification.

Judge the WHOLE selected set together, based on the underlying logic and how much adaptation is realistically needed. Give a confidence 0.0-1.0."""


EXACT_MATCH_SYSTEM_PROMPT = """A client's requirement is already satisfied by one or more features the agency has built (given below). Write a short, plain-language explanation FOR THE CLIENT of why the existing feature(s), or their underlying logic, are an exact match for this requirement.

Describe the existing feature(s) ACCURATELY, exactly as given — do NOT rename them, and do NOT change, add to, or embellish what they actually do just to make them fit the client's requirement. Explain why the real, existing capability already covers what the client asked for.

Keep it to 1-2 sentences, non-technical and reassuring. Return only the explanation, no preamble."""


MODIFY_SET_SYSTEM_PROMPT = """A client's requirement can be met by adapting one or more features the agency already built (given below), but they need some modification.

Explain clearly and concisely WHAT would need to change to adapt the existing feature(s) to fit the requirement. Focus on the practical differences that matter (different domain concepts, extra behaviour needed, data differences). If several features are combined, describe how they fit together. Keep it to 1-3 short sentences, in plain language a non-technical person can follow.

Describe only what to ADD or CHANGE to meet the requirement. Never say that any part of the existing feature(s) we already built is "not needed" or should be removed — frame the adaptation positively, as building on what we have.

If a reviewer's rejection of your previous attempt is included, treat it as instructions and fix exactly what it calls out."""


CRITIC_SYSTEM_PROMPT = """You are reviewing a modification description that explains how an existing software feature (or set of features) would be adapted to meet a client's requirement. You are given the requirement, the existing feature(s), and the proposed modification description.

Judge whether the description is GENUINELY USEFUL: is it concrete and specific about what must change, and does it actually address the gap between the existing feature(s) and the requirement?

Reject it (valid = false) if it is vacuous, empty, generic filler, self-contradictory, merely restates the requirement without saying what to change, or claims the feature is not needed.

Also reject it if the description says any part of the EXISTING feature(s) we already built is "not needed", "not required", or should be removed/discarded. The modification must describe what to ADD or CHANGE to meet the client's requirement — it must never frame the adaptation as trimming down or throwing away what we already built.

When you reject, your reason must say concretely what is missing, so the next attempt can fix it. Otherwise mark it valid."""