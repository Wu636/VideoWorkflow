from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from pathlib import Path
import shutil
import uuid
from src.video_workflow.config import settings

router = APIRouter(prefix="/files", tags=["files"])


class AnalyzeResponse(BaseModel):
    character: str | None = None
    style: str | None = None


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file and return its local absolute path"""
    try:
        # Create uploads directory if it doesn't exist
        upload_dir = settings.OUTPUT_DIR / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename
        ext = Path(file.filename).suffix if file.filename else ".jpg"
        filename = f"{uuid.uuid4()}{ext}"
        file_path = upload_dir / filename
        
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        return {"path": str(file_path.absolute()), "filename": filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_image(file: UploadFile = File(...)):
    """
    Upload and analyze a reference image using multimodal AI.
    Returns character description and visual style.
    """
    try:
        # First save the file
        upload_dir = settings.OUTPUT_DIR / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        ext = Path(file.filename).suffix if file.filename else ".jpg"
        filename = f"{uuid.uuid4()}{ext}"
        file_path = upload_dir / filename
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Now analyze it
        from src.video_workflow.core.analysis import analyze_reference_image
        result = await analyze_reference_image(str(file_path))
        
        if result:
            return AnalyzeResponse(
                character=result.get("character"),
                style=result.get("style")
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to analyze image. Check if GLM or ARK API is configured.")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

