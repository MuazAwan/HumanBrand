# fastapi_backend.py
# Complete FastAPI Backend for Brand Analysis System

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import asyncio
import uuid
import json
import logging
from datetime import datetime
import tempfile
import os
import shutil
from pathlib import Path
import io
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import your existing classes - we'll create an integration bridge
# These imports will be handled by the integration bridge module
from integration_bridge import (
    process_files_and_analyze,
    BrandAnalysisConfig,
    ProcessingResult
)

# In-memory storage (use Redis/Database in production)
job_storage: Dict[str, Dict[str, Any]] = {}
upload_storage: Dict[str, Dict[str, Any]] = {}
active_websockets: Dict[str, WebSocket] = {}

# Application lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Brand Analysis API...")
    
    # Create necessary directories
    temp_dir = Path(tempfile.gettempdir()) / "brand_analysis_uploads"
    temp_dir.mkdir(exist_ok=True)
    
    yield
    
    # Shutdown
    logger.info("Shutting down Brand Analysis API...")
    
    # Cleanup temporary files
    try:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    except Exception as e:
        logger.error(f"Cleanup error: {e}")

# FastAPI app initialization
app = FastAPI(
    title="Brand Analysis API",
    description="AI-Powered Brand Discovery using Bedrock Claude 4 with Advanced Document Processing",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",  # Streamlit default
        "http://0.0.0.0:8501",   # Streamlit docker
        "http://localhost:3000",  # React frontend
        "http://localhost:8080",  # Alternative frontend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================ PYDANTIC MODELS ================

class FileInfo(BaseModel):
    file_id: str
    original_name: str
    size: int
    content_type: str
    upload_timestamp: str

class UploadResponse(BaseModel):
    upload_id: str
    corpus_files: List[FileInfo]
    sitemap_file: Optional[FileInfo] = None
    total_size: int
    message: str
    upload_timestamp: str

class ProcessingOptions(BaseModel):
    use_batch_processing: Optional[bool] = None  # Auto-detect if None
    chunk_size: int = Field(default=8000, ge=1000, le=20000)
    overlap: int = Field(default=800, ge=100, le=2000)
    max_concurrent_chunks: int = Field(default=3, ge=1, le=10)
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    max_tokens: int = Field(default=4000, ge=1000, le=8000)

class AnalysisJobRequest(BaseModel):
    upload_id: str
    processing_options: Optional[ProcessingOptions] = ProcessingOptions()

class AnalysisJobResponse(BaseModel):
    job_id: str
    status: str
    created_at: str
    estimated_duration: Optional[str] = None
    message: str

class AnalysisStep(BaseModel):
    step_number: int
    step_name: str
    description: str
    status: str  # "pending", "processing", "completed", "failed"
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    progress_details: Optional[Dict[str, Any]] = None

class AnalysisStatus(BaseModel):
    job_id: str
    status: str  # "pending", "processing", "completed", "failed"
    overall_progress: int  # 0-100
    current_step: AnalysisStep
    all_steps: List[AnalysisStep]
    processing_method: Optional[str] = None  # "batch" or "traditional"
    estimated_time_remaining: Optional[int] = None
    error_message: Optional[str] = None
    started_at: str
    last_updated: str

class ProcessingMetadata(BaseModel):
    processing_method: str
    total_files: int
    total_chunks: Optional[int] = None
    total_words: int
    total_characters: int
    processing_duration: float
    chunks_processed: Optional[int] = None
    model_settings: Dict[str, Any]

class AnalysisResults(BaseModel):
    job_id: str
    status: str
    objective_data: Optional[Dict[str, Any]] = None
    synthesis_report: Optional[Dict[str, Any]] = None
    processing_metadata: ProcessingMetadata
    agent_results: Dict[str, Any]
    completed_at: str
    download_links: Dict[str, str]

class ErrorResponse(BaseModel):
    error: str
    message: str
    timestamp: str
    job_id: Optional[str] = None

# ================ UTILITY CLASSES ================

class FileManager:
    """Enhanced file management with validation and cleanup"""
    
    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir()) / "brand_analysis_uploads"
        self.temp_dir.mkdir(exist_ok=True)
        self.max_file_size = 50 * 1024 * 1024  # 50MB per file
        self.max_total_size = 200 * 1024 * 1024  # 200MB total
        self.allowed_extensions = {'.txt', '.md', '.docx'}
        self.allowed_mime_types = {
            'text/plain', 
            'text/markdown', 
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        }
    
    def validate_file(self, file: UploadFile) -> None:
        """Validate uploaded file"""
        # Check file size
        if hasattr(file, 'size') and file.size > self.max_file_size:
            raise HTTPException(
                status_code=413,
                detail=f"File {file.filename} is too large. Maximum size: 50MB"
            )
        
        # Check file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in self.allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file extension: {file_ext}. Allowed: {', '.join(self.allowed_extensions)}"
            )
        
        # Check MIME type if available
        if file.content_type and file.content_type not in self.allowed_mime_types:
            logger.warning(f"Unexpected MIME type {file.content_type} for {file.filename}")
    
    async def save_uploaded_files(self, files: List[UploadFile], upload_type: str = "corpus") -> List[FileInfo]:
        """Save uploaded files with validation"""
        saved_files = []
        total_size = 0
        
        for file in files:
            # Validate file
            self.validate_file(file)
            
            # Read file content
            content = await file.read()
            file_size = len(content)
            
            # Check total size limit
            if total_size + file_size > self.max_total_size:
                raise HTTPException(
                    status_code=413,
                    detail=f"Total upload size exceeds limit of 200MB"
                )
            
            # Generate unique filename
            file_id = str(uuid.uuid4())
            safe_filename = f"{file_id}_{upload_type}_{file.filename}"
            file_path = self.temp_dir / safe_filename
            
            # Save file
            with open(file_path, "wb") as f:
                f.write(content)
            
            # Create file info
            file_info = FileInfo(
                file_id=file_id,
                original_name=file.filename,
                size=file_size,
                content_type=file.content_type or "application/octet-stream",
                upload_timestamp=datetime.now().isoformat()
            )
            
            saved_files.append(file_info)
            total_size += file_size
            
            # Store file path mapping
            upload_storage[file_id] = {
                "file_info": file_info,
                "file_path": str(file_path)
            }
            
            logger.info(f"Saved file: {file.filename} ({file_size} bytes) as {safe_filename}")
        
        return saved_files
    
    def get_file_path(self, file_id: str) -> Optional[str]:
        """Get file path by file ID"""
        file_data = upload_storage.get(file_id)
        return file_data["file_path"] if file_data else None
    
    def cleanup_files(self, file_ids: List[str]):
        """Clean up files by file IDs"""
        for file_id in file_ids:
            try:
                file_data = upload_storage.get(file_id)
                if file_data:
                    file_path = file_data["file_path"]
                    if os.path.exists(file_path):
                        os.unlink(file_path)
                    del upload_storage[file_id]
                    logger.info(f"Cleaned up file: {file_path}")
            except Exception as e:
                logger.error(f"Error cleaning up file {file_id}: {e}")

class JobManager:
    """Enhanced job management with detailed progress tracking"""
    
    @staticmethod
    def create_job(job_id: str, initial_data: Dict[str, Any]) -> str:
        """Create a new analysis job"""
        
        # Define analysis steps
        steps = [
            AnalysisStep(
                step_number=1,
                step_name="File Processing",
                description="Processing and validating uploaded files",
                status="pending"
            ),
            AnalysisStep(
                step_number=2,
                step_name="Document Chunking", 
                description="Creating semantic chunks from documents",
                status="pending"
            ),
            AnalysisStep(
                step_number=3,
                step_name="Data Extraction",
                description="Extracting objective data with The Auditor agent",
                status="pending"
            ),
            AnalysisStep(
                step_number=4,
                step_name="Brand Synthesis",
                description="Synthesizing brand analysis with The Archaeologist agent",
                status="pending"
            ),
            AnalysisStep(
                step_number=5,
                step_name="Report Generation",
                description="Generating final reports and downloads",
                status="pending"
            )
        ]
        
        job_data = {
            "job_id": job_id,
            "status": "pending",
            "overall_progress": 0,
            "current_step": steps[0],
            "all_steps": steps,
            "processing_method": None,
            "started_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "error_message": None,
            **initial_data
        }
        
        job_storage[job_id] = job_data
        logger.info(f"Created job {job_id}")
        return job_id
    
    @staticmethod
    def update_job_progress(job_id: str, step_number: int, progress: int, 
                          step_status: str = "processing", details: Optional[Dict[str, Any]] = None):
        """Update job progress for a specific step"""
        if job_id not in job_storage:
            return
        
        job = job_storage[job_id]
        
        # Update current step
        if step_number <= len(job["all_steps"]):
            current_step = job["all_steps"][step_number - 1]
            current_step.status = step_status
            current_step.progress_details = details or {}
            
            if step_status == "processing" and not current_step.started_at:
                current_step.started_at = datetime.now().isoformat()
            elif step_status in ["completed", "failed"]:
                current_step.completed_at = datetime.now().isoformat()
            
            job["current_step"] = current_step
        
        # Update overall progress
        completed_steps = sum(1 for step in job["all_steps"] if step.status == "completed")
        job["overall_progress"] = min(progress, int((completed_steps / len(job["all_steps"])) * 100))
        job["last_updated"] = datetime.now().isoformat()
        
        # Send WebSocket update
        asyncio.create_task(JobManager.send_websocket_update(job_id))
    
    @staticmethod
    def update_job_status(job_id: str, updates: Dict[str, Any]):
        """Update job with arbitrary data"""
        if job_id in job_storage:
            job_storage[job_id].update(updates)
            job_storage[job_id]["last_updated"] = datetime.now().isoformat()
            
            # Send WebSocket update
            asyncio.create_task(JobManager.send_websocket_update(job_id))
    
    @staticmethod
    async def send_websocket_update(job_id: str):
        """Send WebSocket update to connected clients"""
        if job_id in active_websockets and job_id in job_storage:
            try:
                # Convert Pydantic models to dict for JSON serialization
                job_data = job_storage[job_id].copy()
                
                # Convert AnalysisStep objects to dicts
                if "current_step" in job_data:
                    job_data["current_step"] = job_data["current_step"].dict()
                if "all_steps" in job_data:
                    job_data["all_steps"] = [step.dict() for step in job_data["all_steps"]]
                
                await active_websockets[job_id].send_json({
                    "type": "progress_update",
                    "data": job_data,
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"WebSocket send error for job {job_id}: {e}")
                # Remove disconnected WebSocket
                if job_id in active_websockets:
                    del active_websockets[job_id]
    
    @staticmethod
    def complete_job(job_id: str, results: Dict[str, Any]):
        """Mark job as completed with results"""
        if job_id in job_storage:
            # Mark all steps as completed
            for step in job_storage[job_id]["all_steps"]:
                if step.status != "completed":
                    step.status = "completed"
                    step.completed_at = datetime.now().isoformat()
            
            job_storage[job_id].update({
                "status": "completed",
                "overall_progress": 100,
                "completed_at": datetime.now().isoformat(),
                **results
            })
            
            # Send final WebSocket update
            asyncio.create_task(JobManager.send_websocket_update(job_id))
    
    @staticmethod
    def fail_job(job_id: str, error_message: str, step_number: Optional[int] = None):
        """Mark job as failed"""
        if job_id in job_storage:
            if step_number and step_number <= len(job_storage[job_id]["all_steps"]):
                job_storage[job_id]["all_steps"][step_number - 1].status = "failed"
                job_storage[job_id]["all_steps"][step_number - 1].completed_at = datetime.now().isoformat()
            
            job_storage[job_id].update({
                "status": "failed",
                "error_message": error_message,
                "failed_at": datetime.now().isoformat()
            })
            
            # Send failure WebSocket update
            asyncio.create_task(JobManager.send_websocket_update(job_id))
    
    @staticmethod
    def get_job(job_id: str) -> Optional[Dict[str, Any]]:
        """Get job data"""
        return job_storage.get(job_id)

# Initialize managers
file_manager = FileManager()

# ================ API ENDPOINTS ================

@app.get("/", tags=["Root"])
async def root():
    """API root endpoint with service information"""
    return {
        "service": "Brand Analysis API",
        "version": "2.0.0",
        "status": "running",
        "description": "AI-Powered Brand Discovery using Bedrock Claude 4",
        "features": [
            "Multi-file document processing",
            "Intelligent batch processing",
            "Real-time progress tracking",
            "WebSocket updates",
            "Smart semantic chunking",
            "Advanced brand synthesis"
        ],
        "endpoints": {
            "upload": "/api/v1/upload-files",
            "analyze": "/api/v1/start-analysis",
            "status": "/api/v1/analysis/{job_id}",
            "results": "/api/v1/results/{job_id}",
            "download": "/api/v1/download/{job_id}/{file_type}",
            "websocket": "/ws/analysis/{job_id}",
            "health": "/health",
            "docs": "/docs"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/v1/upload-files", response_model=UploadResponse, tags=["File Management"])
async def upload_files(
    corpus_files: List[UploadFile] = File(..., description="Multiple corpus files (.md, .txt, .docx)"),
    sitemap_file: Optional[UploadFile] = File(None, description="Optional sitemap file")
):
    """
    Upload multiple corpus files and optional sitemap file for brand analysis.
    
    - **corpus_files**: List of document files containing brand content
    - **sitemap_file**: Optional sitemap document for context
    
    Returns upload ID and file information for starting analysis.
    """
    try:
        logger.info(f"Uploading {len(corpus_files)} corpus files")
        
        # Validate minimum requirements
        if not corpus_files:
            raise HTTPException(status_code=400, detail="At least one corpus file is required")
        
        if len(corpus_files) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 corpus files allowed")
        
        # Save corpus files
        corpus_file_infos = await file_manager.save_uploaded_files(corpus_files, "corpus")
        
        # Save sitemap file if provided
        sitemap_file_info = None
        if sitemap_file:
            sitemap_file_infos = await file_manager.save_uploaded_files([sitemap_file], "sitemap")
            sitemap_file_info = sitemap_file_infos[0] if sitemap_file_infos else None
        
        # Generate upload ID
        upload_id = str(uuid.uuid4())
        
        # Calculate total size
        total_size = sum(file_info.size for file_info in corpus_file_infos)
        if sitemap_file_info:
            total_size += sitemap_file_info.size
        
        # Store upload metadata
        upload_storage[upload_id] = {
            "upload_id": upload_id,
            "corpus_files": [info.dict() for info in corpus_file_infos],
            "sitemap_file": sitemap_file_info.dict() if sitemap_file_info else None,
            "total_size": total_size,
            "upload_timestamp": datetime.now().isoformat()
        }
        
        response = UploadResponse(
            upload_id=upload_id,
            corpus_files=corpus_file_infos,
            sitemap_file=sitemap_file_info,
            total_size=total_size,
            message=f"Successfully uploaded {len(corpus_files)} corpus files" + 
                   (f" and 1 sitemap file" if sitemap_file else ""),
            upload_timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Upload completed: {upload_id}, Total size: {total_size} bytes")
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File upload error: {e}")
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

@app.post("/api/v1/start-analysis", response_model=AnalysisJobResponse, tags=["Analysis"])
async def start_analysis(
    request: AnalysisJobRequest,
    background_tasks: BackgroundTasks
):
    """
    Start brand analysis with uploaded files.
    
    Initiates the complete brand analysis pipeline including:
    - Document processing and chunking
    - Data extraction with The Auditor agent
    - Brand synthesis with The Archaeologist agent
    - Report generation
    """
    try:
        # Validate upload ID
        if request.upload_id not in upload_storage:
            raise HTTPException(status_code=404, detail="Upload ID not found")
        
        upload_data = upload_storage[request.upload_id]
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Estimate duration based on file size
        total_size = upload_data["total_size"]
        estimated_minutes = max(2, min(15, total_size // (1024 * 1024)))  # 2-15 minutes based on size
        
        # Create job record
        JobManager.create_job(job_id, {
            "upload_id": request.upload_id,
            "processing_options": request.processing_options.dict(),
            "estimated_duration": f"{estimated_minutes} minutes"
        })
        
        # Start background processing
        background_tasks.add_task(
            process_brand_analysis_background, 
            job_id, 
            request.upload_id, 
            request.processing_options
        )
        
        response = AnalysisJobResponse(
            job_id=job_id,
            status="pending",
            created_at=datetime.now().isoformat(),
            estimated_duration=f"{estimated_minutes} minutes",
            message="Brand analysis started successfully. Use WebSocket or polling to track progress."
        )
        
        logger.info(f"Analysis started: job_id={job_id}, upload_id={request.upload_id}")
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis start error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start analysis: {str(e)}")

@app.get("/api/v1/analysis/{job_id}", response_model=AnalysisStatus, tags=["Analysis"])
async def get_analysis_status(job_id: str):
    """
    Get detailed analysis job status including step-by-step progress.
    
    Returns comprehensive status information including:
    - Overall progress percentage
    - Current processing step
    - All step statuses
    - Processing method (batch vs traditional)
    - Time estimates and error information
    """
    job = JobManager.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return AnalysisStatus(
        job_id=job["job_id"],
        status=job["status"],
        overall_progress=job["overall_progress"],
        current_step=job["current_step"],
        all_steps=job["all_steps"],
        processing_method=job.get("processing_method"),
        estimated_time_remaining=job.get("estimated_time_remaining"),
        error_message=job.get("error_message"),
        started_at=job["started_at"],
        last_updated=job["last_updated"]
    )

@app.get("/api/v1/results/{job_id}", response_model=AnalysisResults, tags=["Analysis"])
async def get_analysis_results(job_id: str):
    """
    Get complete analysis results for a completed job.
    
    Returns the full brand analysis including:
    - Objective data extracted by The Auditor
    - Synthesis report from The Archaeologist  
    - Processing metadata and performance metrics
    - Download links for reports and data
    """
    job = JobManager.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Analysis not completed. Current status: {job['status']}"
        )
    
    # Generate download links
    download_links = {
        "markdown_report": f"/api/v1/download/{job_id}/markdown",
        "json_data": f"/api/v1/download/{job_id}/json",
        "full_results": f"/api/v1/download/{job_id}/full"
    }
    
    return AnalysisResults(
        job_id=job["job_id"],
        status=job["status"],
        objective_data=job.get("objective_data"),
        synthesis_report=job.get("synthesis_report"),
        processing_metadata=job.get("processing_metadata", {}),
        agent_results=job.get("agent_results", {}),
        completed_at=job.get("completed_at"),
        download_links=download_links
    )

@app.get("/api/v1/download/{job_id}/{file_type}", tags=["Downloads"])
async def download_results(job_id: str, file_type: str):
    """
    Download analysis results in various formats.
    
    - **markdown**: Download the brand analysis report as Markdown
    - **json**: Download structured data as JSON
    - **full**: Download complete results as JSON
    """
    job = JobManager.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Analysis not completed")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    try:
        if file_type == "markdown":
            content = job.get("synthesis_report", {}).get("report_markdown", "")
            if not content:
                raise HTTPException(status_code=404, detail="Markdown report not found")
            
            return StreamingResponse(
                io.StringIO(content),
                media_type="text/markdown",
                headers={
                    "Content-Disposition": f"attachment; filename=brand_analysis_{timestamp}.md"
                }
            )
        
        elif file_type == "json":
            objective_data = job.get("objective_data", {})
            if not objective_data:
                raise HTTPException(status_code=404, detail="JSON data not found")
            
            content = json.dumps(objective_data, indent=2)
            return StreamingResponse(
                io.StringIO(content),
                media_type="application/json",
                headers={
                    "Content-Disposition": f"attachment; filename=objective_data_{timestamp}.json"
                }
            )
        
        elif file_type == "full":
            full_results = {
                "job_id": job_id,
                "objective_data": job.get("objective_data", {}),
                "synthesis_report": job.get("synthesis_report", {}),
                "processing_metadata": job.get("processing_metadata", {}),
                "agent_results": job.get("agent_results", {}),
                "completed_at": job.get("completed_at")
            }
            
            content = json.dumps(full_results, indent=2)
            return StreamingResponse(
                io.StringIO(content),
                media_type="application/json",
                headers={
                    "Content-Disposition": f"attachment; filename=full_results_{timestamp}.json"
                }
            )
        
        else:
            raise HTTPException(status_code=400, detail="Invalid file type. Use: markdown, json, or full")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download error: {e}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

@app.websocket("/ws/analysis/{job_id}")
async def websocket_analysis_updates(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint for real-time analysis progress updates.
    
    Provides live updates including:
    - Step-by-step progress
    - Processing status changes
    - Error notifications
    - Completion notifications
    """
    await websocket.accept()
    active_websockets[job_id] = websocket
    
    try:
        # Send initial status
        job = JobManager.get_job(job_id)
        if job:
            # Convert objects to dicts for JSON serialization
            job_data = job.copy()
            if "current_step" in job_data:
                job_data["current_step"] = job_data["current_step"].dict()
            if "all_steps" in job_data:
                job_data["all_steps"] = [step.dict() for step in job_data["all_steps"]]
            
            await websocket.send_json({
                "type": "initial_status",
                "data": job_data,
                "timestamp": datetime.now().isoformat()
            })
        
        # Keep connection alive and handle client messages
        while True:
            try:
                # Wait for client ping or disconnect
                message = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                
                # Handle client ping
                if message == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    })
            
            except asyncio.TimeoutError:
                # Send heartbeat if no client activity
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": datetime.now().isoformat()
                })
            
            except WebSocketDisconnect:
                break
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for job {job_id}")
    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {e}")
    finally:
        if job_id in active_websockets:
            del active_websockets[job_id]

@app.get("/health", tags=["System"])
async def health_check():
    """System health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "active_jobs": len(job_storage),
        "active_websockets": len(active_websockets),
        "temp_files": len(upload_storage),
        "system_info": {
            "temp_dir_exists": file_manager.temp_dir.exists(),
            "temp_dir_size": sum(f.stat().st_size for f in file_manager.temp_dir.glob('*') if f.is_file()) if file_manager.temp_dir.exists() else 0
        }
    }

@app.delete("/api/v1/cleanup/{upload_id}", tags=["File Management"])
async def cleanup_upload(upload_id: str):
    """Clean up uploaded files and associated data"""
    try:
        if upload_id not in upload_storage:
            raise HTTPException(status_code=404, detail="Upload ID not found")
        
        upload_data = upload_storage[upload_id]
        
        # Collect file IDs to clean up
        file_ids = []
        if "corpus_files" in upload_data:
            for file_info in upload_data["corpus_files"]:
                if isinstance(file_info, dict) and "file_id" in file_info:
                    file_ids.append(file_info["file_id"])
        
        if upload_data.get("sitemap_file") and "file_id" in upload_data["sitemap_file"]:
            file_ids.append(upload_data["sitemap_file"]["file_id"])
        
        # Clean up files
        file_manager.cleanup_files(file_ids)
        
        # Remove upload data
        del upload_storage[upload_id]
        
        return {
            "message": f"Successfully cleaned up upload {upload_id}",
            "files_removed": len(file_ids),
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

# ================ BACKGROUND PROCESSING ================

async def process_brand_analysis_background(
    job_id: str, 
    upload_id: str, 
    processing_options: ProcessingOptions
):
    """
    Background task for processing brand analysis using your existing logic
    """
    try:
        logger.info(f"Starting background analysis for job {job_id}")
        
        # Get upload data
        if upload_id not in upload_storage:
            JobManager.fail_job(job_id, f"Upload ID {upload_id} not found")
            return
        
        upload_data = upload_storage[upload_id]
        
        # Step 1: File Processing
        JobManager.update_job_progress(job_id, 1, 10, "processing", {
            "message": "Processing uploaded files"
        })
        
        # Prepare file paths for processing
        corpus_file_paths = []
        for file_info in upload_data["corpus_files"]:
            file_id = file_info["file_id"] if isinstance(file_info, dict) else file_info.file_id
            file_path = file_manager.get_file_path(file_id)
            if file_path and os.path.exists(file_path):
                corpus_file_paths.append(file_path)
            else:
                JobManager.fail_job(job_id, f"File not found: {file_id}", 1)
                return
        
        sitemap_path = None
        if upload_data.get("sitemap_file"):
            sitemap_info = upload_data["sitemap_file"]
            file_id = sitemap_info["file_id"] if isinstance(sitemap_info, dict) else sitemap_info.file_id
            sitemap_path = file_manager.get_file_path(file_id)
        
        JobManager.update_job_progress(job_id, 1, 25, "completed", {
            "files_processed": len(corpus_file_paths),
            "sitemap_included": sitemap_path is not None
        })
        
        # Step 2: Document Processing and Chunking
        JobManager.update_job_progress(job_id, 2, 30, "processing", {
            "message": "Creating semantic chunks from documents"
        })
        
        # Use the integration bridge to process files
        config = BrandAnalysisConfig(
            use_batch_processing=processing_options.use_batch_processing,
            chunk_size=processing_options.chunk_size,
            overlap=processing_options.overlap,
            max_concurrent_chunks=processing_options.max_concurrent_chunks,
            temperature=processing_options.temperature,
            max_tokens=processing_options.max_tokens
        )
        
        # Progress callback for integration bridge
        def progress_callback(step: int, progress: int, message: str, details: Dict[str, Any] = None):
            JobManager.update_job_progress(job_id, step, progress, "processing", {
                "message": message,
                **(details or {})
            })
        
        # Run the analysis through integration bridge
        try:
            results = await process_files_and_analyze(
                file_paths=corpus_file_paths,
                sitemap_path=sitemap_path,
                config=config,
                progress_callback=progress_callback
            )
            
            # Step 5: Finalization
            JobManager.update_job_progress(job_id, 5, 95, "processing", {
                "message": "Finalizing results and generating downloads"
            })
            
            # Process results
            processing_metadata = ProcessingMetadata(
                processing_method=results.processing_method,
                total_files=results.total_files,
                total_chunks=results.total_chunks,
                total_words=results.total_words,
                total_characters=results.total_characters,
                processing_duration=results.processing_duration,
                chunks_processed=results.chunks_processed,
                model_settings=results.model_settings
            )
            
            # Complete the job
            JobManager.complete_job(job_id, {
                "objective_data": results.objective_data,
                "synthesis_report": results.synthesis_report,
                "processing_metadata": processing_metadata.dict(),
                "agent_results": results.agent_results
            })
            
            logger.info(f"Analysis completed successfully for job {job_id}")
        
        except Exception as analysis_error:
            logger.error(f"Analysis error for job {job_id}: {analysis_error}")
            JobManager.fail_job(job_id, f"Analysis failed: {str(analysis_error)}")
    
    except Exception as e:
        logger.error(f"Background processing error for job {job_id}: {e}")
        JobManager.fail_job(job_id, f"Processing failed: {str(e)}")

# ================ ERROR HANDLERS ================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=f"HTTP {exc.status_code}",
            message=exc.detail,
            timestamp=datetime.now().isoformat()
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            message="An unexpected error occurred",
            timestamp=datetime.now().isoformat()
        ).dict()
    )

# ================ STARTUP AND SHUTDOWN ================

@app.on_event("startup")
async def startup_event():
    """Additional startup tasks"""
    logger.info("Brand Analysis API startup complete")
    logger.info(f"Temp directory: {file_manager.temp_dir}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Brand Analysis API...")
    
    # Close WebSocket connections
    for job_id in list(active_websockets.keys()):
        try:
            await active_websockets[job_id].close()
        except:
            pass
    
    # Cleanup temporary files
    try:
        for file_id in list(upload_storage.keys()):
            file_data = upload_storage[file_id]
            if "file_path" in file_data and os.path.exists(file_data["file_path"]):
                os.unlink(file_data["file_path"])
    except Exception as e:
        logger.error(f"Cleanup error: {e}")

if __name__ == "__main__":
    import uvicorn
    
    # Development configuration
    uvicorn.run(
        "fastapi_backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )