from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import whisper
import uuid
import os

app = FastAPI()

# ✅ Enable CORS (allow any origin during development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all origins
    allow_credentials=True,
    allow_methods=["*"],   # allow all HTTP methods
    allow_headers=["*"],   # allow all headers
)

# Load Whisper model once (change "base" → "large-v3" for best Japanese accuracy)
model = whisper.load_model("base")

# Job storage
JOBS_DIR = "jobs"
os.makedirs(JOBS_DIR, exist_ok=True)

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """Handle short audio clips (<=30s). Returns result immediately."""
    try:
        uuid_str = str(uuid.uuid4())
        audio_path = f"temp_{uuid_str}_{file.filename}"
        with open(audio_path, "wb") as f:
            f.write(await file.read())

        result = model.transcribe(audio_path, language="ja")
        os.remove(audio_path)
        return {"text": result["text"]}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

def process_long_audio(file_path: str, job_id: str):
    """Background worker for long audio files."""
    result = model.transcribe(file_path, language="ja")
    with open(os.path.join(JOBS_DIR, f"{job_id}.txt"), "w", encoding="utf-8") as f:
        f.write(result["text"])
    os.remove(file_path)

@app.post("/transcribe-long")
async def transcribe_long(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Start background transcription for long audio. Returns job_id."""
    try: 
        job_id = str(uuid.uuid4())
        file_path = os.path.join(JOBS_DIR, f"{job_id}_{file.filename}")
        with open(file_path, "wb") as f:
            f.write(await file.read())
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    background_tasks.add_task(process_long_audio, file_path, job_id)
    return {"job_id": job_id, "status": "processing"}

@app.get("/result/{job_id}")
async def get_result(job_id: str):
    """Check if a long transcription is finished."""
    try:
        result_path = os.path.join(JOBS_DIR, f"{job_id}.txt")
        if os.path.exists(result_path):
            with open(result_path, "r", encoding="utf-8") as f:
                return {"job_id": job_id, "status": "done", "text": f.read()}
        else:
            return {"job_id": job_id, "status": "processing"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
