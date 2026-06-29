
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

#Only for Windows, to avoid DLL load errors for CUDA (Sewen Laptop)
#os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin")

from contextlib import asynccontextmanager
from typing import Optional

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
    PORT: int = 8000  # Different from Next.js (3000) and Postgres (5432)


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
    
    if "CUDAExecutionProvider" not in available_providers:
        logger.warning("CUDAExecutionProvider not available! Falling back to CPU.")
        providers = ["CPUExecutionProvider"]
    else:
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
# ENDPOINTS
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint for Docker/Kubernetes probes."""
    return {
        "status": "healthy",
        "model_loaded": face_app is not None,
        "gpu_memory_limit_mb": Config.GPU_MEMORY_LIMIT_MB
    }


@app.post("/register", response_model=RegisterResponse)
async def register_student(file: UploadFile = File(...)):
    """
    Register a student by extracting face embedding from a headshot.
    
    Workflow:
    1. Receive uploaded image (single high-quality headshot)
    2. Detect face (expect exactly 1 face)
    3. Extract 512-dimensional embedding
    4. Return embedding as JSON for Next.js to save to PostgreSQL
    
    Args:
        file: Uploaded image file (JPEG/PNG)
    
    Returns:
        RegisterResponse with embedding array
    
    Raises:
        400: No face detected, or multiple faces detected
    """
    logger.info(f"Processing /register request: {file.filename}")
    
    # --- Read and decode image ---
    image_bytes = await file.read()
    image = decode_image(image_bytes)
    
    # --- Downscale if needed (headshots usually don't need this, but safety check) ---
    image = downscale_image(image, Config.MAX_IMAGE_WIDTH)
    
    # --- Detect faces ---
    faces = face_app.get(image)
    
    if len(faces) == 0:
        raise HTTPException(
            status_code=400,
            detail="No face detected in the image. Please upload a clear headshot."
        )
    
    if len(faces) > 1:
        logger.warning(f"Multiple faces detected ({len(faces)}). Using the largest face.")
        # Select the face with the largest bounding box area
        faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    
    # --- Extract embedding ---
    # InsightFace returns embedding as numpy array (512,)
    embedding = faces[0].embedding
    
    # Convert to list for JSON serialization
    embedding_list = embedding.tolist()
    
    logger.info(f"Successfully extracted embedding for student. Shape: {embedding.shape}")
    
    return RegisterResponse(
        embedding=embedding_list,
        face_count=len(faces)
    )


@app.post("/process-attendance", response_model=DetectResponse)
async def detect_attendance(
    file: UploadFile = File(...),
    students_json: str = Form(...)
):
    """
    Detect attendance from a classroom photo.
    
    Workflow:
    1. Receive uploaded classroom photo + list of registered students (with embeddings)
    2. Downscale image to prevent VRAM overflow
    3. Detect all faces in the photo
    4. Extract embeddings for each detected face
    5. Compare against registered students using cosine similarity
    6. Return list of matched student IDs
    
    Args:
        file: Uploaded classroom photo (wide-angle)
        students_json: JSON string of student list, format:
            [{"student_id": "S001", "embedding": [0.1, 0.2, ...]}, ...]
    
    Returns:
        DetectResponse with matched student IDs
    
    Raises:
        400: Invalid JSON, no faces detected, or no students provided
    """
    logger.info(f"Processing /detect request: {file.filename}")
    
    # --- Parse students JSON ---
    try:
        students_data = json.loads(students_json)
        students = [StudentEmbedding(**s) for s in students_data]
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in students_json: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing students data: {str(e)}")
    
    if len(students) == 0:
        raise HTTPException(status_code=400, detail="No students provided for matching")
    
    logger.info(f"Matching against {len(students)} registered students")
    
    # --- Read and decode image ---
    image_bytes = await file.read()
    image = decode_image(image_bytes)
    
    # --- CRITICAL: Downscale to prevent VRAM overflow ---
    image = downscale_image(image, Config.MAX_IMAGE_WIDTH)
    
    # --- Detect all faces in classroom photo ---
    faces = face_app.get(image)
    
    if len(faces) == 0:
        logger.warning("No faces detected in classroom photo")
        return DetectResponse(
            matched_student_ids=[],
            total_faces_detected=0,
            similarity_threshold=Config.SIMILARITY_THRESHOLD
        )
    
    logger.info(f"Detected {len(faces)} faces in classroom photo")
    
    # --- Extract embeddings from all detected faces ---
    detected_embeddings = np.array([face.embedding for face in faces])  # Shape: (N, 512)
    
    # --- Prepare gallery embeddings from registered students ---
    gallery_embeddings = np.array([s.embedding for s in students])  # Shape: (M, 512)
    student_ids = [s.student_id for s in students]
    
    # --- Match faces to students ---
    matched_student_ids = set()  # Use set to avoid duplicates
    
    for i, detected_emb in enumerate(detected_embeddings):
        # Compute cosine similarity between this face and all students
        similarities = cosine_similarity_matrix(detected_emb, gallery_embeddings)
        
        # Find the best match
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[best_match_idx]
        
        # Check if similarity exceeds threshold
        if best_similarity >= Config.SIMILARITY_THRESHOLD:
            matched_id = student_ids[best_match_idx]
            matched_student_ids.add(matched_id)
            logger.debug(
                f"Face {i+1} matched to {matched_id} "
                f"(similarity: {best_similarity:.3f})"
            )
        else:
            logger.debug(
                f"Face {i+1} not matched "
                f"(best similarity: {best_similarity:.3f} < {Config.SIMILARITY_THRESHOLD})"
            )
    
    logger.info(f"Matched {len(matched_student_ids)} students out of {len(faces)} detected faces")
    
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