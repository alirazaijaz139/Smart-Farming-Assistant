from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import auth, farms, crop
from .routers import auth, farms, crop, disease
from .routers import auth, farms, crop, disease, rag






from .database import Base, engine
from .routers import auth, farms

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Smart Farming Assistant", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict to your Vercel domain before deploying
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(farms.router)
app.include_router(crop.router)
app.include_router(disease.router)
app.include_router(rag.router)


@app.get("/")
def health():
    return {"status": "ok", "app": "Smart Farming Assistant"}