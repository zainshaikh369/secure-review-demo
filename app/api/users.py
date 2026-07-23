from fastapi import APIRouter

router = APIRouter()

USERS = [
    {"id": 1, "name": "Zain", "role": "admin"},
    {"id": 2, "name": "Alice", "role": "user"},
]


@router.get("/")
def list_users() -> list[dict]:
    # Intentionally missing authentication and authorization.
    return USERS


@router.get("/{user_id}")
def get_user(user_id: int) -> dict:
    for user in USERS:
        if user["id"] == user_id:
            return user

    return {"error": "User not found"}