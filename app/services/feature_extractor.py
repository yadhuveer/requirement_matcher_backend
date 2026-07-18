import asyncio
import json
from anthropic import AsyncAnthropic
from app.config import settings

client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)

EXTRACTION_SYSTEM_PROMPT = """You are analyzing documentation for a software project that has already been built and delivered. Your goal is to extract the distinct FEATURES that were implemented, and describe each one in a way that captures its MEANING in plain, non-technical language.
 
This is important: these descriptions will later be matched against how NEW CLIENTS describe what they want. Clients describe outcomes in everyday language ("a way for students to be grouped and given lessons"), NOT in technical terms ("JWT auth", "CRUD"). So each feature's description must capture what the feature DOES FOR USERS, in plain language, with enough context to be understood on its own.
 
For each feature, produce an object with exactly these fields:
- "name": a short human label for the feature (3-6 words).
- "description": a self-contained, plain-language sentence describing what the feature does for its users and WHY it matters. NO technical jargon. Include the domain context (e.g. "delivery vehicle", "student", "invoice") so the meaning is unambiguous. This is the most important field.
- "domain": the business area this feature belongs to (e.g. "logistics", "authentication", "payments", "education", "reporting").
- "tech_details": any specific technical implementation details mentioned (e.g. "JWT with refresh tokens", "Razorpay integration"). Use an empty string if none are given. This field is for internal reference only.
 
Rules:
- Each feature object must represent exactly ONE capability — a single thing a client could ask for, build, or remove on its own. If one paragraph or section describes several capabilities that could each be requested independently, output a SEPARATE feature object for each one. Do NOT bundle multiple independently-requestable capabilities into a single feature just because the document describes them together in the same sentence or section.
- To decide if a piece of text is one capability or several, apply this test: "Could a client plausibly want one of these without the other?" If yes, they are separate features and must each get their own object. If no, they belong together as one feature.
- Do NOT over-split. Small individual actions that only make sense as part of a single capability (the create/edit/delete variations of managing one kind of thing) together form ONE feature. The right level of granularity is "a distinct thing a client would ask for as a single line item", not every individual action or button.
- Extract every distinct feature you can find, whether the document is neatly structured with headings or written as continuous unstructured prose.
- The "description" must make sense to a non-technical business person reading it cold, with no surrounding context.
- Preserve the DOMAIN in the description so the meaning is unambiguous on its own.
- Ignore project background, team size, timelines, pricing, and client names. Only actual functionality.
- Do not invent features that aren't supported by the text.
- If the section contains no features, return an empty array.
 
Return ONLY a JSON array of these objects. No markdown, no explanation, no preamble."""


async def extract_features_from_chunk(chunk: str) -> list[dict]:
    """Send ONE chunk to Claude and get back a list of rich feature objects."""
    response = await client.messages.create(
        model=settings.LLM_MODEL,
        max_tokens=3000,
        temperature=0,
        system=EXTRACTION_SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": f"Document section:\n---\n{chunk}\n---"}
        ],
    )

    raw = response.content[0].text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        features = json.loads(raw)
        if not isinstance(features, list):
            return []
        clean = []
        for f in features:
            if isinstance(f, dict) and f.get("description", "").strip():
                clean.append({
                    "name": str(f.get("name", "")).strip(),
                    "description": str(f["description"]).strip(),
                    "domain": str(f.get("domain", "")).strip(),
                    "tech_details": str(f.get("tech_details", "")).strip(),
                })
        return clean
    except json.JSONDecodeError:
        return []


async def extract_features_from_chunks(chunks: list[str]) -> list[dict]:
    """Extract from all chunks, at most `max_concurrent` at a time."""
    max_concurrent = 3
    semaphore = asyncio.Semaphore(max_concurrent)

    async def extract_with_limit(chunk: str) -> list[dict]:
        async with semaphore:
            return await extract_features_from_chunk(chunk)

    tasks = [extract_with_limit(chunk) for chunk in chunks]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_features = []
    for result in results:
        if isinstance(result, Exception):
            continue
        all_features.extend(result)

    return all_features