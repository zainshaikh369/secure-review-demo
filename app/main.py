from fastapi import FastAPI

from app.api.auth import router as auth_router
from app.api.users import router as users_router

app = FastAPI(
    title="Secure Review Demo",
    version="0.1.0",
)

app.include_router(auth_router, prefix="/auth", tags=["authentication"])
app.include_router(users_router, prefix="/users", tags=["users"])


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "healthy"}