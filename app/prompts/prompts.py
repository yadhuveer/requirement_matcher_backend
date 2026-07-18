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