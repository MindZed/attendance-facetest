
"""
MindZed Technologies - Classroom Attendance AI Microservice
===========================================================
FastAPI service for face detection and recognition using InsightFace.

Architecture:
  - Next.js PWA  <->  PostgreSQL (pgvector)  <->  THIS SERVICE
  - Endpoints: /register (single headshot -> embedding)
               /detect   (classroom photo -> matched student IDs)

Author: Abdul Kadir
"""

import io
import json
import os
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

# Only execute this block if the operating system is Windows ('nt')
if os.name == 'nt':
    cuda_path = os.getenv("CUDA_DLL_PATH")
    if cuda_path and os.path.exists(cuda_path):
        try:
            os.add_dll_directory(cuda_path)
            print(f"🔧 Windows CUDA DLL Path injected: {cuda_path}")
        except Exception as e:
            print(f"⚠️ Failed to add CUDA DLL directory: {e}")
            #add the path to merged CUDA/cuDNN

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from insightface.app import FaceAnalysis
from loguru import logger
from pydantic import BaseModel, Field


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Centralized configuration with explicit 4GB VRAM safeguards."""
    
    # --- VRAM & Memory Constraints ---
    # RTX 3050 has 4GB VRAM. We reserve ~500MB for OS/display buffer
    # and set ONNX Runtime's hard limit to 3.5GB.
    GPU_MEMORY_LIMIT_MB: int = 3500
    
    # --- Image Preprocessing ---
    # Classroom photos are downscaled to this max width before detection.
    # 1920px is sufficient for face detection and prevents OOM on wide images.
    MAX_IMAGE_WIDTH: int = 1920
    
    # InsightFace detection resolution (internal). Lower = faster + less VRAM.
    # 640x640 is a good balance for classroom photos. Use 320x320 if OOM persists.
    DETECTION_SIZE: tuple = (640, 640)
    
    # --- Model Selection ---
    # 'buffalo_l' = highest accuracy, ~550MB VRAM for models
    # 'buffalo_s' = smaller, ~300MB VRAM, slightly less accurate
    # We use 'buffalo_l' with memory limits. Switch to 'buffalo_s' if needed.
    MODEL_NAME: str = "buffalo_l"
    
    # --- Face Matching ---
    # Cosine similarity threshold. InsightFace embeddings work well at 0.4+.
    # Higher = stricter (fewer false positives, more false negatives)
    SIMILARITY_THRESHOLD: float = 0.4
    
    # --- Server ---
    HOST: str = "0.0.0.0"
    port_env = os.getenv("PORT", "8000")
    PORT: int = int(port_env) # Different from Next.js (3000) and Postgres (5432)


# ============================================================================
# GLOBAL STATE
# ============================================================================

# InsightFace app instance (loaded once at startup)
face_app: Optional[FaceAnalysis] = None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def downscale_image(image: np.ndarray, max_width: int) -> np.ndarray:
    """
    Downscale image to max_width while preserving aspect ratio.
    Uses INTER_AREA for downscaling (better quality than INTER_LINEAR).
    
    Args:
        image: OpenCV BGR image (H, W, C)
        max_width: Maximum allowed width in pixels
    
    Returns:
        Resized image (or original if already within limit)
    """
    height, width = image.shape[:2]
    
    if width <= max_width:
        return image
    
    scale = max_width / width
    new_height = int(height * scale)
    
    logger.debug(f"Downscaling image from {width}x{height} to {max_width}x{new_height}")
    
    # INTER_AREA is optimal for downscaling (avoids aliasing)
    resized = cv2.resize(image, (max_width, new_height), interpolation=cv2.INTER_AREA)
    return resized


def decode_image(image_bytes: bytes) -> np.ndarray:
    """
    Decode image bytes (from upload) to OpenCV BGR format.
    Supports JPEG, PNG, WebP.
    
    Raises:
        HTTPException: If image cannot be decoded
    """
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    
    # Decode with OpenCV
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image format. Supported: JPEG, PNG, WebP"
        )
    
    return image


def cosine_similarity_matrix(query_embedding: np.ndarray, gallery_embeddings: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a single query embedding and multiple gallery embeddings.
    Vectorized for performance.
    
    Args:
        query_embedding: Shape (512,)
        gallery_embeddings: Shape (N, 512)
    
    Returns:
        Similarity scores, shape (N,)
    """
    # Normalize embeddings (L2 norm)
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    gallery_norms = gallery_embeddings / np.linalg.norm(gallery_embeddings, axis=1, keepdims=True)
    
    # Dot product = cosine similarity for normalized vectors
    similarities = np.dot(gallery_norms, query_norm)
    
    return similarities


def warm_up_model():
    """
    Run a dummy inference to warm up the model and allocate VRAM.
    This prevents first-request latency and ensures VRAM is properly allocated.
    """
    logger.info("Warming up model with dummy inference...")
    
    # Create a dummy image (640x640 BGR)
    dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
    
    # Run detection (will be slow on first run, but allocates VRAM)
    faces = face_app.get(dummy_image)
    
    logger.info(f"Model warm-up complete. Detected {len(faces)} faces in dummy image (expected 0).")


# ============================================================================
# LIFESPAN (STARTUP & SHUTDOWN)
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle:
    - Startup: Load model, configure GPU memory, warm up
    - Shutdown: Clean up resources
    """
    global face_app
    
    logger.info("=" * 60)
    logger.info("Starting MindZed AI Microservice")
    logger.info("=" * 60)
    
    # --- Configure ONNX Runtime GPU Memory ---
    logger.info(f"Configuring ONNX Runtime GPU memory limit: {Config.GPU_MEMORY_LIMIT_MB}MB")
    
    # Set environment variable for ONNX Runtime (must be set before session creation)
    os.environ["ORT_GPU_MEM_LIMIT_IN_MB"] = str(Config.GPU_MEMORY_LIMIT_MB)
    
    # Verify CUDA is available
    available_providers = ort.get_available_providers()
    logger.info(f"Available ONNX Runtime providers: {available_providers}")
    
    # --- NEW TOGGLE LOGIC ---
    target = os.getenv("AI_HARDWARE_TARGET", "gpu").lower()

    if target == "cpu":
        logger.info("💻 AI Core: Hardware target explicitly set to CPU in .env")
        providers = ["CPUExecutionProvider"]
    elif "CUDAExecutionProvider" not in available_providers:
        logger.warning("⚠️ CUDA requested but not found! Falling back to CPU.")
        providers = ["CPUExecutionProvider"]
    else:
        logger.info("🤖 AI Core: Target set to NVIDIA GPU")
        # Convert MB to Bytes for the ONNX config
        gpu_mem_limit_bytes = Config.GPU_MEMORY_LIMIT_MB * 1024 * 1024
        
        # Explicitly configure the CUDA provider to strictly obey the 4GB limit
        providers = [
            ("CUDAExecutionProvider", {
                "device_id": 0,
                "gpu_mem_limit": gpu_mem_limit_bytes,
                "arena_extend_strategy": "kNextPowerOfTwo",
            }),
            "CPUExecutionProvider"
        ]
    
    # --- Load InsightFace Model ---
    logger.info(f"Loading InsightFace model: {Config.MODEL_NAME}")
    
    face_app = FaceAnalysis(
        name=Config.MODEL_NAME,
        providers=providers,
        allowed_modules=["detection", "recognition"]  # Skip age/gender to save VRAM
    )
    
    # Prepare the model with detection size
    face_app.prepare(ctx_id=0, det_size=Config.DETECTION_SIZE)
    
    logger.info("Model loaded successfully")
    
    # --- Warm Up ---
    warm_up_model()
    
    logger.info("=" * 60)
    logger.info("Service ready to accept requests")
    logger.info("=" * 60)
    
    yield  # Application runs here
    
    # --- Shutdown ---
    logger.info("Shutting down AI microservice...")
    # InsightFace doesn't have explicit cleanup, but we can log it
    logger.info("Cleanup complete")


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="MindZed Attendance AI",
    description="Facial recognition microservice for classroom attendance",
    version="1.0.0",
    lifespan=lifespan
)


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class StudentEmbedding(BaseModel):
    """Represents a registered student with their face embedding."""
    student_id: str = Field(..., description="Unique student identifier")
    embedding: list[float] = Field(..., description="512-dimensional face embedding")


class RegisterResponse(BaseModel):
    """Response for /register endpoint."""
    embedding: list[float] = Field(..., description="512-dimensional face embedding")
    face_count: int = Field(..., description="Number of faces detected (should be 1)")


class DetectResponse(BaseModel):
    """Response for /detect endpoint."""
    matched_student_ids: list[str] = Field(..., description="List of matched student IDs")
    total_faces_detected: int = Field(..., description="Total faces found in classroom photo")
    similarity_threshold: float = Field(..., description="Threshold used for matching")


# ============================================================================
# TEMPORARY LOCAL DB HELPERS (For Postman Testing)
# ============================================================================
DB_FILE = "local_db.json"

def load_local_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            return json.load(f)
    return []

def save_local_db(data):
    with open(DB_FILE, "w") as f:
        json.dump(data, f)

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.post("/register", response_model=RegisterResponse)
async def register_student(
    student_id: str = Form(...), 
    file: UploadFile = File(...)
):
    """Extracts embedding and saves it directly to local_db.json."""
    logger.info(f"Processing /register request for: {student_id}")
    
    image_bytes = await file.read()
    image = decode_image(image_bytes)
    image = downscale_image(image, Config.MAX_IMAGE_WIDTH)
    
    faces = face_app.get(image)
    if len(faces) == 0:
        raise HTTPException(status_code=400, detail="No face detected.")
    
    embedding = faces[0].embedding.tolist()
    
    # --- Local DB Save Logic ---
    db_data = load_local_db()
    db_data.append({
        "student_id": student_id,
        "embedding": embedding
    })
    save_local_db(db_data)
    
    return RegisterResponse(embedding=embedding, face_count=len(faces))


@app.post("/process-attendance", response_model=DetectResponse)
async def detect_attendance(file: UploadFile = File(...)):
    """Reads students from local_db.json and matches against the uploaded classroom photo."""
    logger.info(f"Processing /process-attendance request")
    
    # --- Load from Local DB instead of form data ---
    students_data = load_local_db()
    if not students_data:
        raise HTTPException(status_code=400, detail="Database is empty. Register students first.")
        
    students = [StudentEmbedding(**s) for s in students_data]
    
    image_bytes = await file.read()
    image = decode_image(image_bytes)
    image = downscale_image(image, Config.MAX_IMAGE_WIDTH)
    
    faces = face_app.get(image)
    if len(faces) == 0:
        return DetectResponse(matched_student_ids=[], total_faces_detected=0, similarity_threshold=Config.SIMILARITY_THRESHOLD)
    
    detected_embeddings = np.array([face.embedding for face in faces])
    gallery_embeddings = np.array([s.embedding for s in students])
    student_ids = [s.student_id for s in students]
    
    matched_student_ids = set()
    for i, detected_emb in enumerate(detected_embeddings):
        similarities = cosine_similarity_matrix(detected_emb, gallery_embeddings)
        best_match_idx = np.argmax(similarities)
        if similarities[best_match_idx] >= Config.SIMILARITY_THRESHOLD:
            matched_student_ids.add(student_ids[best_match_idx])
            
    return DetectResponse(
        matched_student_ids=list(matched_student_ids),
        total_faces_detected=len(faces),
        similarity_threshold=Config.SIMILARITY_THRESHOLD
    )


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler to prevent VRAM leaks and provide clean errors."""
    logger.exception(f"Unhandled exception: {exc}")
    
    # Attempt to clear GPU cache (if using PyTorch backend, but good practice)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass  # PyTorch not installed, skip
    
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check logs for details."}
    )


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting server on {Config.HOST}:{Config.PORT}")
    
    uvicorn.run(
        "main:app",
        host=Config.HOST,
        port=Config.PORT,
        reload=False,  # Disable reload in production
        workers=1,     # Single worker to prevent VRAM conflicts
        log_level="info"
    )