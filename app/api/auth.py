from fastapi import APIRouter

router = APIRouter()


@router.post("/login")
def login(username: str, password: str) -> dict[str, str]:
    # Intentionally insecure for the code-review demo.
    if username == "admin" and password == "admin123":
        return {"token": "demo-admin-token"}

    return {"error": "Invalid credentials"}