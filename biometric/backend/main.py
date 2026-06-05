"""
main.py
FastAPI backend — biometric face recognition system.
Endpoints: register, authenticate, users, logs, delete
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import logging

from database import (
    init_db, register_user, get_all_embeddings,
    log_access, get_all_users, get_access_logs, delete_user
)
from face_engine import extract_embedding, find_match

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# Initialize DB on startup
init_db()

app = FastAPI(
    title="Biometric Face Recognition System",
    description="Student/Employee identification via facial recognition",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ────────────────────────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    name:       str
    student_id: str
    role:       str = "student"
    department: Optional[str] = None
    image_b64:  str   # base64 encoded image from webcam


class AuthRequest(BaseModel):
    image_b64: str   # base64 encoded image from webcam


class DeleteRequest(BaseModel):
    student_id: str


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "system": "Biometric Face Recognition System",
        "version": "1.0.0",
        "status": "running",
        "endpoints": ["/register", "/authenticate", "/users", "/logs", "/delete"]
    }


@app.get("/health")
def health():
    users = get_all_users()
    return {"status": "healthy", "registered_users": len(users)}


@app.post("/register")
def register(req: RegisterRequest):
    """
    Register a new user with their face.
    Accepts base64 image, extracts FaceNet embedding, stores in DB.
    """
    log.info(f"Registering: {req.name} ({req.student_id})")

    # Extract face embedding
    result = extract_embedding(req.image_b64)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])

    # Store in database
    db_result = register_user(
        name=req.name,
        student_id=req.student_id,
        role=req.role,
        department=req.department,
        embedding=result["embedding"]
    )

    if not db_result["success"]:
        raise HTTPException(status_code=409, detail=db_result["message"])

    return {
        "status":  "success",
        "message": db_result["message"],
        "user_id": db_result["user_id"]
    }


@app.post("/authenticate")
def authenticate(req: AuthRequest):
    """
    Authenticate a user from webcam image.
    Extracts embedding → compares against all registered users → returns match.
    """
    log.info("Authentication attempt...")

    # Extract probe embedding
    result = extract_embedding(req.image_b64)
    if not result["success"]:
        log_access(None, None, None, "ERROR", 0)
        raise HTTPException(status_code=400, detail=result["error"])

    # Load all registered embeddings
    registered = get_all_embeddings()

    # Find best match
    match = find_match(result["embedding"], registered)

    if match["matched"]:
        user = match["user"]
        log_access(
            user_id=user["user_id"],
            student_id=user["student_id"],
            name=user["name"],
            status="GRANTED",
            confidence=match["confidence"]
        )
        return {
            "status":     "GRANTED",
            "name":       user["name"],
            "student_id": user["student_id"],
            "role":       user["role"],
            "department": user["department"],
            "confidence": match["confidence"],
            "message":    f"Welcome, {user['name']}!"
        }
    else:
        log_access(None, None, "UNKNOWN", "DENIED", match.get("confidence", 0))
        return {
            "status":     "DENIED",
            "message":    "Face not recognised. Access denied.",
            "confidence": match.get("confidence", 0),
        }


@app.get("/users")
def list_users():
    """Return all registered users."""
    return {"users": get_all_users(), "total": len(get_all_users())}


@app.get("/logs")
def access_logs(limit: int = 50):
    """Return recent access logs."""
    return {"logs": get_access_logs(limit), "total": limit}


@app.delete("/delete")
def remove_user(req: DeleteRequest):
    """Delete a user and their face embedding."""
    result = delete_user(req.student_id)
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result["message"])
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
