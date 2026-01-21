from fastapi import APIRouter, HTTPException, BackgroundTasks
from pathlib import Path
import json
import logging
from typing import Optional

from src.video_workflow.core.orchestrator import WorkflowOrchestrator
from src.video_workflow.config import settings
from src.video_workflow.types import Storyboard, Scene
from src.video_workflow.server.schemas import (
    CreateSessionRequest, SessionResponse, ReviseScriptRequest,
    GenerateImagesRequest, GenerateVideosRequest
)

router = APIRouter(prefix="/sessions", tags=["sessions"])
logger = logging.getLogger(__name__)

# Single orchestrator instance for now (could be per-request if needed)
orchestrator = WorkflowOrchestrator()

def get_session_path(session_id: str) -> Path:
    path = settings.OUTPUT_DIR / session_id
    if not path.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    return path

def load_storyboard(session_id: str) -> Storyboard:
    session_dir = get_session_path(session_id)
    script_file = session_dir / "script.json"
    if not script_file.exists():
        raise HTTPException(status_code=404, detail="Script not found")
    
    try:
        with open(script_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return Storyboard(**data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load script: {str(e)}")

def save_storyboard(session_id: str, storyboard: Storyboard):
    session_dir = get_session_path(session_id)
    script_file = session_dir / "script.json"
    with open(script_file, "w", encoding="utf-8") as f:
        f.write(storyboard.model_dump_json(indent=2))

@router.post("", response_model=SessionResponse)
async def create_session(request: CreateSessionRequest):
    """Start a new session: generate script from topic"""
    try:
        # We need to run orchestration steps.
        # Run step 1: Generate Storyboard
        # Note: orchestrator.llm.generate_storyboard returns a Storyboard object
        # but orchestrator.run is designed to run the full flow or parts and return session_path.
        # We should use orchestrator methods granualarly.
        
        await orchestrator.initialize()
        
        # 1. Generate Storyboard
        storyboard = await orchestrator.llm.generate_storyboard(
            topic=request.topic,
            count=request.count,
            reference_image=request.reference_image,  # TODO: Handle base64 or path?
            # analyze_reference_image usually takes a path. 
            # If request.reference_image is base64, we might need to save it first?
            # For now assume it logic inside llm handles it or we need to save it.
            template=request.template,
            include_dialogue=request.include_dialogue
        )
        
        # Create session directory manually since we are breaking down steps
        import time
        session_id = str(int(time.time()))
        session_dir = settings.OUTPUT_DIR / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Save reference image if provided (to keep session self-contained)
        # Note: logic for handling base64 vs path needs care. 
        # For this prototype, assume logic handles path or we'd need to upload separate.
        # Simplest: Save script.
        
        with open(session_dir / "script.json", "w") as f:
            f.write(storyboard.model_dump_json(indent=2))
            
        return SessionResponse(
            session_id=session_id,
            status="SCRIPT_GENERATED",
            storyboard=storyboard
        )
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{session_id}/script", response_model=Storyboard)
async def get_script(session_id: str):
    return load_storyboard(session_id)

@router.post("/{session_id}/script/revise", response_model=Storyboard)
async def revise_script(session_id: str, request: ReviseScriptRequest):
    storyboard = load_storyboard(session_id)
    try:
        new_storyboard = await orchestrator.revise_storyboard(
            storyboard, 
            request.feedback,
            request.reference_image
        )
        save_storyboard(session_id, new_storyboard)
        return new_storyboard
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{session_id}/script", response_model=Storyboard)
async def update_script(session_id: str, storyboard: Storyboard):
    # Manual update
    save_storyboard(session_id, storyboard)
    return storyboard

@router.post("/{session_id}/images")
async def generate_images(session_id: str, request: GenerateImagesRequest):
    storyboard = load_storyboard(session_id)
    session_dir = get_session_path(session_id)
    
    try:
        # Run image generation
        # orchestrator.run_image_generation saves images to session_dir/images
        path, success = await orchestrator.run_image_generation(
            storyboard,
            reference_image=request.reference_image,
            existing_session_dir=str(session_dir),
            scene_ids=request.scene_ids
        )
        # Reload storyboard to get updated image paths (run_image_generation updates paths in memory? 
        # It writes to script.json at end, so we should reload)
        updated_storyboard = load_storyboard(session_id)
        return {"status": "completed", "success": success, "storyboard": updated_storyboard}
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{session_id}/videos")
async def generate_videos(session_id: str, request: GenerateVideosRequest):
    storyboard = load_storyboard(session_id)
    session_dir = get_session_path(session_id)
    
    try:
        await orchestrator.run_video_generation(
            storyboard,
            str(session_dir),
            scene_ids=request.scene_ids
        )
        # Concatenate? Maybe separate endpoint or auto?
        # For now just generate clips.
        
        updated_storyboard = load_storyboard(session_id)
        return {"status": "completed", "storyboard": updated_storyboard}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{session_id}/concatenate")
async def concatenate_videos(session_id: str):
    """Concatenate all scene videos into a single final video"""
    from src.video_workflow.core.video_processing import concatenate_videos as concat_videos
    
    storyboard = load_storyboard(session_id)
    session_dir = get_session_path(session_id)
    
    try:
        # Collect all video paths
        video_files = []
        for scene in storyboard.scenes:
            if scene.video_path:
                video_path = Path(scene.video_path)
                if not video_path.is_absolute():
                    video_path = Path(settings.OUTPUT_DIR) / scene.video_path.replace("outputs/", "")
                if video_path.exists():
                    video_files.append(video_path)
        
        if not video_files:
            raise HTTPException(status_code=400, detail="No videos to concatenate")
        
        # Concatenate videos
        final_video_path = concat_videos(str(session_dir), video_files)
        
        # Return relative path
        relative_path = str(Path(final_video_path).relative_to(settings.OUTPUT_DIR))
        
        return {
            "status": "completed",
            "final_video_path": f"outputs/{relative_path}"
        }
    except Exception as e:
        logger.error(f"Video concatenation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/download/{session_id}/{filename}")
async def download_file(session_id: str, filename: str):
    """Download a file with proper Content-Disposition header"""
    from fastapi.responses import FileResponse
    
    session_dir = get_session_path(session_id)
    
    # Security: only allow downloading from specific subdirectories
    if filename == "final_video.mp4":
        file_path = session_dir / filename
    elif filename.startswith("scene_"):
        # Extract scene number and look up from script
        try:
            scene_id = int(filename.replace("scene_", "").replace(".mp4", ""))
            storyboard = load_storyboard(session_id)
            scene = next((s for s in storyboard.scenes if s.id == scene_id), None)
            if not scene or not scene.video_path:
                raise HTTPException(status_code=404, detail="Video not found")
            
            video_path = Path(scene.video_path)
            if not video_path.is_absolute():
                file_path = Path(settings.OUTPUT_DIR) / scene.video_path.replace("outputs/", "")
            else:
                file_path = video_path
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid filename")
    else:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=str(file_path),
        media_type="video/mp4",
        filename=filename,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )
