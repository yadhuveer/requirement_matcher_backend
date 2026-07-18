from openai import AsyncOpenAI
from app.config import settings

client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

EMBEDDING_MODEL = "text-embedding-3-small"


async def embed_texts(texts: list[str]) ->list[list[float]]:
    if not texts:
        return []
    
    response = await client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )


    sorted_items = sorted(response.data, key=lambda item: item.index)
    return [item.embedding for item in sorted_items]


async def embed_features(features: list[dict]) -> list[dict]:
    if  not features:
        return []
    
    descriptions = [f["description"] for f in features]
    vectors = await embed_texts(descriptions)

    result = []

    for feature, vector in zip(features, vectors):
        result.append({**feature, "embedding":vector })
    return result
