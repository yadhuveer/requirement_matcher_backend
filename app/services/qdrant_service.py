from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from app.config import settings


client = QdrantClient(
    url=settings.QDRANT_ENDPOINT,
    api_key=settings.QDRANT_API_KEY,
    timeout=60
)

COLLECTION_NAME = "completed_features"
VECTOR_SIZE = 1536         


def ensure_collection():
   
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,   
            ),
        )


def upsert_features(embedded_features: list[dict], project_id: int):
    
    points = []
    for f in embedded_features:
        points.append(
            PointStruct(
                id=f["id"],                         
                vector=f["embedding"],
                payload={
                    "feature_id": f["id"],
                    "project_id": project_id,
                    "name": f.get("name", ""),
                    "description": f.get("description", ""),
                    "domain": f.get("domain", ""),
                    "tech_details": f.get("tech_details", ""),
                },
            )
        )

    # Upsert in batches so a large feature set doesn't become one giant write
    # that times out against the (distant) Qdrant Cloud region.
    BATCH_SIZE = 64
    for start in range(0, len(points), BATCH_SIZE):
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points[start:start + BATCH_SIZE],
        )
    return len(points)