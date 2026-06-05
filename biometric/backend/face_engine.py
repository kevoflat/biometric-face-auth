"""
face_engine.py - insightface (ArcFace/ONNX). No dlib/TF required.
"""
import numpy as np, base64, cv2, logging
from io import BytesIO
from PIL import Image

log = logging.getLogger(__name__)
COSINE_THRESHOLD = 0.35


def _get_app():
    if not hasattr(_get_app, "_app"):
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(320, 320))
        _get_app._app = app
    return _get_app._app


def base64_to_bgr(b64):
    if "," in b64:
        b64 = b64.split(",")[1]
    img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def extract_embedding(image_b64):
    try:
        app = _get_app()
        bgr = base64_to_bgr(image_b64)

        # Resize large images — cuts inference time significantly
        h, w = bgr.shape[:2]
        if max(h, w) > 640:
            scale = 640 / max(h, w)
            bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)))

        faces = app.get(bgr)

        if not faces:
            return {"success": False, "error": "No face detected. Ensure face is clearly visible."}
        if len(faces) > 1:
            return {"success": False, "error": f"{len(faces)} faces detected. Use one face only."}

        emb = faces[0].embedding
        emb = emb / np.linalg.norm(emb)
        return {"success": True, "embedding": emb.tolist()}

    except Exception as e:
        log.error(f"Embedding error: {e}")
        return {"success": False, "error": str(e)}


def find_match(probe_embedding, registered_users):
    if not registered_users:
        return {"matched": False, "reason": "No users registered yet", "confidence": 0}

    known = np.array([u["embedding"] for u in registered_users])
    probe = np.array(probe_embedding)
    scores   = np.dot(known, probe)
    best_idx = int(np.argmax(scores))
    best_sim = float(scores[best_idx])
    confidence = round(min(100.0, max(0.0, best_sim * 100)), 2)

    if best_sim >= COSINE_THRESHOLD:
        return {"matched": True, "user": registered_users[best_idx],
                "confidence": confidence, "score": round(best_sim, 4)}
    return {"matched": False, "confidence": confidence,
            "reason": f"No match (best: {best_sim:.3f}, threshold: {COSINE_THRESHOLD})"}