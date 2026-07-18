import asyncio
import json
from anthropic import AsyncAnthropic
from app.config import settings
 
client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
 

DEDUPE_SYSTEM_PROMPT = """You are given a list of software features extracted from different sections of the SAME project document. Because the sections overlapped, the list may contain TRUE DUPLICATES: the exact same capability extracted more than once, described in slightly different words.
 
Your ONLY job is to remove those true duplicates. This is a duplicate-removal task, NOT a summarization or consolidation task.
 
DEFAULT BEHAVIOUR: KEEP FEATURES SEPARATE. When in doubt, do NOT merge. It is far better to leave two similar features separate than to wrongly combine two distinct capabilities. A later step matches these features against what new clients ask for, and clients ask for capabilities individually — so each distinct capability must remain its own feature.
 
ONLY merge two features when they describe the SAME single capability — i.e. one is clearly just a reworded re-extraction of the other, produced because the document sections overlapped. If you removed one, a reader would lose NO information, because the other says the same thing.
 
Do NOT merge two features if each one could be requested, built, or removed independently of the other. If a client could plausibly ask for one WITHOUT the other, they are DIFFERENT features and must stay separate — even if they are related, complementary, part of the same workflow, or commonly sold together. When unsure whether two features are the same capability or two related capabilities, KEEP THEM SEPARATE.
 
Test to apply to any pair before merging: "Are these two descriptions saying the same thing twice, or are they two different things that happen to be related?" Only the first case is a duplicate. The second must stay separate.
 
If a single entry actually bundles two or more independently-requestable capabilities together, SPLIT it into separate features.
 
When you do merge a true duplicate:
- Keep the single most complete, plain-language, self-contained description.
- Pick the single most appropriate domain.
- Combine tech_details (comma-separated, no duplication).
 
Each output feature keeps exactly these fields: "name", "description", "domain", "tech_details".
 
Return ONLY a JSON array of the cleaned feature objects. No markdown, no explanation, no preamble."""
 
 
async def dedupe_features(features: list[dict]) -> list[dict]:
    """One LLM pass over the whole list to merge duplicates and near-duplicates."""
    if len(features) <= 1:
        return features
 
    features_json = json.dumps(features, ensure_ascii=False, indent=2)
 
    response = await client.messages.create(
        model=settings.LLM_MODEL,
        max_tokens=8000,
        temperature=0,
        system=DEDUPE_SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": f"Feature list:\n{features_json}"}
        ],
    )
 
    raw = response.content[0].text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
 
    try:
        cleaned = json.loads(raw)
        if not isinstance(cleaned, list) or not cleaned:
            return features   # fall back to raw list rather than losing everything
        result = []
        for f in cleaned:
            if isinstance(f, dict) and f.get("description", "").strip():
                result.append({
                    "name": str(f.get("name", "")).strip(),
                    "description": str(f["description"]).strip(),
                    "domain": str(f.get("domain", "")).strip(),
                    "tech_details": str(f.get("tech_details", "")).strip(),
                })
        return result if result else features
    except json.JSONDecodeError:
        return features       # same graceful fallback