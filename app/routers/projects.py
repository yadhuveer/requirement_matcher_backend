import time
from fastapi import APIRouter, UploadFile, File, Form, HTTPException,status,Depends
from typing import Optional
from app.services.pdf_service import extract_pages,build_chunks
from app.services.extraction_agent import extract_features_from_chunks
from fastapi.concurrency import run_in_threadpool
from app.services.dedupe import dedupe_features
from app.services.project_service import create_project_with_features
from app.database import get_db
from sqlalchemy.orm import Session
from app.services.embedding_service import embed_features
from app.services.qdrant_service import ensure_collection, upsert_features
from app.database import SessionLocal
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
    t = time.perf_counter()
    raw_features = await extract_features_from_chunks(chunks)
    print(f"[timing] extraction : {time.perf_counter() - t:.1f}s  ({len(raw_features)} raw features)")

    t = time.perf_counter()
    features = await dedupe_features(raw_features)
    print(f"[timing] dedupe     : {time.perf_counter() - t:.1f}s  ({len(features)} after dedupe)")
    

    if not features:
        raise HTTPException(400, "No features could be extracted from this document.")
    

    
    # The request-scoped `db` connection has sat idle during the minutes-long
    # extraction, so Neon may have dropped it ("SSL connection has been closed
    # unexpectedly"). Use a FRESH, pre-pinged session for the DB write so the
    # save doesn't fail on a dead connection.
    t = time.perf_counter()
    write_db = SessionLocal()
    try:
        result = create_project_with_features(
            write_db, project_name, client_name, contact_info, features, created_by=current_user.id
        )
    finally:
        write_db.close()
    print(f"[timing] db save    : {time.perf_counter() - t:.1f}s")

    t = time.perf_counter()
    embedded = await embed_features(result["features"])
    print(f"[timing] embed      : {time.perf_counter() - t:.1f}s")

    t = time.perf_counter()
    ensure_collection()
    await run_in_threadpool(upsert_features, embedded, result["project_id"])
    print(f"[timing] qdrant     : {time.perf_counter() - t:.1f}s")

    return {
        "project_id": result["project_id"],
        "feature_count": len(result["features"]),
    }