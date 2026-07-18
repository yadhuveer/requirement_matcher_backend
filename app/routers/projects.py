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
from app.services.auth_dependency import get_current_user
from app.models.db_models import User


router = APIRouter(prefix="/api/extract")

@router.post("/feature")
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
    features = await dedupe_features(raw_features)
    print(raw_features)
    print("--------")
    print(features)

    if not features:
        raise HTTPException(400, "No features could be extracted from this document.")
    

    
    result = create_project_with_features(
        db, project_name, client_name, contact_info, features,created_by=current_user.id
    )

    

    embedded = await embed_features(result["features"])


    ensure_collection()
    count = await run_in_threadpool(upsert_features, embedded, result["project_id"])
    print(count)