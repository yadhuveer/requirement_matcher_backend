from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers.auth import router as userRouter
from app.routers.projects import router as projectRouter
from app.routers.requirements import router as requirementRouter

app= FastAPI()

app.add_middleware(
    CORSMiddleware,
    
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "https://requirement-matcher-frontend.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(userRouter)
app.include_router(projectRouter)
app.include_router(requirementRouter)

@app.get("/health")
def health_Check():
    return {"status":"ok", "message":"Requirement Matcher AOI is running"}