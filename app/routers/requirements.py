from fastapi import APIRouter, UploadFile, File, Form, HTTPException,status,Depends
from typing import Optional
from app.services.pdf_service import extract_pages,build_chunks
from app.services.feature_extractor import extract_features_from_chunks
from fastapi.concurrency import run_in_threadpool
from app.services.dedupe import dedupe_features
from app.services.project_service import create_project_with_features
from app.database import get_db
from sqlalchemy.orm import Session
from app.services.embedding_service import embed_features
from app.services.qdrant_service import ensure_collection, upsert_features
from sqlalchemy.orm import Session
from app.database import get_db
import asyncio
from app.services.matching_graph import match_requirement
from app.services.requirement_service import save_requirement_results
from app.services.auth_dependency import get_current_user
from app.models.db_models import User

router = APIRouter(prefix="/api/analyse")

@router.post("/requirement")
async def extractFeatures(file:UploadFile=File(...),project_name:str=Form(...),client_name:Optional[str]=Form(None),contact_info:Optional[str]=Form(None),db: Session = Depends(get_db),current_user: User = Depends(get_current_user)):
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are allowed"
        )
    
    pdf_bytes = await file.read()

    MAX_SIZE = 5*1024*1024

    if len(pdf_bytes)>MAX_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File exceeds the 5MB limit."
        )

    pages = await run_in_threadpool(extract_pages,pdf_bytes)

    chunks = build_chunks(pages)
    raw_features = await extract_features_from_chunks(chunks)
    requirements = await dedupe_features(raw_features)

    results = await asyncio.gather(
        *[match_requirement(req) for req in requirements]
    )

    
    saved = save_requirement_results(
        db, project_name, client_name, contact_info, list(results),user_id=current_user.id
    )

    return {
        "project_id": saved["project_id"],
        "requirement_count": len(results),
        "results": results,
    }