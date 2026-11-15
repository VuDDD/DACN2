import os
import shutil
import uuid
from typing import Any, List, Dict
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from fastapi.responses import FileResponse
from PIL import Image

from app.core.config import settings
from app.storage.service import storage_service, ALLOWED_EXTENSIONS
from app.models.user import User
from app.api.deps import get_current_user

router = APIRouter(prefix="/images", tags=["Image Management"])

@router.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Upload a retinal photograph (JPG, PNG, TIF) and validate dimensions."""
    relative_url, abs_path = await storage_service.save_upload(file, subfolder="uploads")
    
    # Extract resolution and metadata
    try:
        with Image.open(abs_path) as img:
            width, height = img.size
            mode = img.mode
            format_name = img.format
    except Exception as e:
        if os.path.exists(abs_path):
            os.remove(abs_path)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Uploaded file is corrupted or not a valid medical image: {e}"
        )

    return {
        "url": relative_url,
        "filename": os.path.basename(abs_path),
        "width": width,
        "height": height,
        "mode": mode,
        "format": format_name,
        "size_bytes": os.path.getsize(abs_path)
    }

@router.get("/serve/{subfolder}/{filename}")
def serve_image(subfolder: str, filename: str):
    """Serve image, mask, or PDF file directly to client."""
    # Sanitize path to prevent directory traversal
    clean_sub = os.path.basename(subfolder)
    clean_fn = os.path.basename(filename)
    abs_path = os.path.join(storage_service.local_root, clean_sub, clean_fn)

    if not os.path.exists(abs_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")

    ext = os.path.splitext(clean_fn)[1].lower()
    media_type = "image/jpeg"
    if ext == ".png":
        media_type = "image/png"
    elif ext in (".tif", ".tiff"):
        media_type = "image/tiff"
    elif ext == ".pdf":
        media_type = "application/pdf"
    elif ext == ".gif":
        media_type = "image/gif"

    return FileResponse(
        path=abs_path,
        media_type=media_type,
        headers={"Cache-Control": "public, max-age=86400"}
    )

@router.get("/samples")
def list_sample_images() -> List[Dict[str, Any]]:
    """Lists preloaded real benchmark images from IDRiD and DRIVE for rapid testing."""
    samples = []
    
    # IDRiD Testing set samples
    idrid_dir = os.path.join("data", "IDRiD", "1. Original Images", "b. Testing Set")
    if os.path.exists(idrid_dir):
        files = sorted([f for f in os.listdir(idrid_dir) if f.lower().endswith(".jpg")])
        for f in files[:6]:
            samples.append({
                "dataset": "IDRiD",
                "filename": f,
                "label": f"IDRiD Test: {f}",
                "task_type": "IDRID_LESIONS",
                "description": "Indian Diabetic Retinopathy Image Dataset - Lesions (MA, HE, EX, SE, OD)"
            })

    # DRIVE Test set samples
    drive_dir = os.path.join("data", "DRIVE", "test", "images")
    if os.path.exists(drive_dir):
        files = sorted([f for f in os.listdir(drive_dir) if f.lower().endswith((".tif", ".png", ".jpg"))])
        for f in files[:4]:
            samples.append({
                "dataset": "DRIVE",
                "filename": f,
                "label": f"DRIVE Test: {f}",
                "task_type": "DRIVE_VESSELS",
                "description": "Digital Retinal Images for Vessel Extraction - Blood Vessels"
            })

    return samples

@router.post("/load-sample")
def load_sample_into_storage(
    dataset: str,
    filename: str,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Copies a dataset benchmark sample into storage to use for analysis."""
    clean_fn = os.path.basename(filename)
    if dataset == "IDRiD":
        source_path = os.path.join("data", "IDRiD", "1. Original Images", "b. Testing Set", clean_fn)
        task_type = "IDRID_LESIONS"
    elif dataset == "DRIVE":
        source_path = os.path.join("data", "DRIVE", "test", "images", clean_fn)
        task_type = "DRIVE_VESSELS"
    else:
        raise HTTPException(status_code=400, detail="Unknown dataset")

    if not os.path.exists(source_path):
        raise HTTPException(status_code=404, detail="Sample image not found on disk")

    ext = os.path.splitext(clean_fn)[1]
    unique_fn = f"sample_{uuid.uuid4().hex[:8]}_{clean_fn}"
    target_dir = os.path.join(storage_service.local_root, "uploads")
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, unique_fn)

    shutil.copyfile(source_path, target_path)

    with Image.open(target_path) as img:
        w, h = img.size

    return {
        "url": f"/api/images/serve/uploads/{unique_fn}",
        "filename": unique_fn,
        "width": w,
        "height": h,
        "task_type": task_type
    }

