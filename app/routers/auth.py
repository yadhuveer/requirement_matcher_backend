import os

from fastapi import APIRouter, Depends, HTTPException, status, Response
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.db_models import User
from app.models.userschemas import UserCreate, UserLogin, UserResponse, Token
from app.services.auth_utils import hash_password, verify_password, create_access_token


# In production (frontend + backend on different domains) the auth cookie is
# cross-site, so it needs SameSite=None + Secure. Locally (both on localhost)
# it is same-site over http, so SameSite=Lax + Secure=False is required instead.
IS_PROD = os.getenv("ENVIRONMENT") == "production"
COOKIE_SECURE = IS_PROD
COOKIE_SAMESITE = "none" if IS_PROD else "lax"


router = APIRouter(prefix="/api/user", tags=["users"])

@router.post("/signup", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def signup(payload:UserCreate, db:Session=Depends(get_db)):
    existing = db.query(User).filter(payload.email==User.email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    hashed = hash_password(payload.password)

    new_user = User(name=payload.name,email=payload.email,password_hash=hashed)
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return new_user

@router.post("/login")
def login(payload:UserLogin, response: Response, db:Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    if not verify_password(payload.password,user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    token = create_access_token(data={"sub":str(user.id), "email":user.email})

    
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE,
        max_age=60 * 60 * 24,   
        path="/",
    )

    return {"message": "Login successful"}