"""Protected async SCEPT inference API. Uploaded MRI is temporary; no raw data logging."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import secrets
import shutil
import tempfile
import threading
import time
from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from .model import run

app=FastAPI(title="BrainMirror SCEPT live backend",version="1.0")
ORIGIN=os.getenv("SCEPT_WEB_ORIGIN","https://brainmirror-observatory.vercel.app").rstrip("/")
app.add_middleware(CORSMiddleware,allow_origins=[ORIGIN],allow_methods=["GET","POST"],
                   allow_headers=["Authorization","Content-Type"])
SECRET=os.getenv("SCEPT_RUN_TOKEN","")
POOL=ThreadPoolExecutor(max_workers=1,thread_name_prefix="scept")
JOBS={}
LOCK=threading.RLock()
MAX_FILE=90*1024*1024


def require_auth(value):
    if not SECRET:raise HTTPException(503,"Inference service has no run token configured")
    supplied=value[7:] if isinstance(value,str) and value.startswith("Bearer ") else ""
    if not secrets.compare_digest(supplied,SECRET):
        raise HTTPException(401,"Private SCEPT access token required")


@app.middleware("http")
async def reject_private_endpoint_without_auth(request, call_next):
    # Authenticate before Starlette parses large multipart uploads.
    if request.url.path.startswith("/api/v1/runs") and request.method!="OPTIONS":
        try:
            require_auth(request.headers.get("authorization"))
        except HTTPException as e:
            return JSONResponse({"detail":e.detail},status_code=e.status_code)
    return await call_next(request)


@app.get("/health")
def health():
    import torch
    return {"status":"ready" if SECRET else "not_configured",
            "service":"scept_live","cuda":torch.cuda.is_available(),
            "authentication":"required","mode":"real_pretrained_model"}


async def upload_one(file:UploadFile,name:str,folder:Path):
    filename=file.filename or ""
    if not filename.lower().endswith((".nii",".nii.gz")):
        raise HTTPException(400,name+" must be a .nii or .nii.gz NIfTI")
    dest=folder/(name+(".nii.gz" if filename.lower().endswith(".gz") else ".nii"))
    total=0
    try:
        with dest.open("wb") as stream:
            while True:
                data=await file.read(1024*1024)
                if not data:break
                total+=len(data)
                if total>MAX_FILE:raise HTTPException(413,"MRI file exceeds 90 MiB limit")
                stream.write(data)
        if not total:raise HTTPException(400,"Empty MRI upload")
    finally:
        await file.close()
    return dest


def execute(id,paths,folder):
    def progress(step):
        with LOCK:
            if id in JOBS:JOBS[id].update(step=step,updated=time.time())
    try:
        payload=run(paths,progress=progress)
        with LOCK:
            JOBS[id].update(status="complete",step="complete",result=payload,updated=time.time())
    except Exception as exc:
        # Never return raw paths, potentially identifying filenames, or stack traces.
        with LOCK:
            JOBS[id].update(status="failed",step="failed",
                            error_type=type(exc).__name__,updated=time.time())
    finally:
        shutil.rmtree(folder,ignore_errors=True)


@app.post("/api/v1/runs",status_code=202)
async def submit(flair:UploadFile=File(...),t1ce:UploadFile=File(...),
                 t1:UploadFile=File(...),t2:UploadFile=File(...),
                 authorization:str|None=Header(default=None)):
    require_auth(authorization)
    with LOCK:
        for old,job in list(JOBS.items()):
            if time.time()-job["updated"]>3600:JOBS.pop(old,None)
        if any(j["status"] in ("queued","running") for j in JOBS.values()):
            raise HTTPException(429,"Another model experiment is in progress")
        if len(JOBS)>=20:raise HTTPException(429,"Job registry capacity reached")
    folder=Path(tempfile.mkdtemp(prefix="scept-user-"))
    try:
        paths={}
        for key,file in (("flair",flair),("t1ce",t1ce),("t1",t1),("t2",t2)):
            paths[key]=await upload_one(file,key,folder)
    except Exception:
        shutil.rmtree(folder,ignore_errors=True)
        raise
    id=secrets.token_urlsafe(24)
    with LOCK:JOBS[id]={"job_id":id,"status":"queued","step":"queued",
                        "result":None,"updated":time.time()}
    POOL.submit(execute,id,paths,folder)
    with LOCK:
        if JOBS[id]["status"]=="queued":JOBS[id]["status"]="running"
    return {"job_id":id,"status":"running","poll_path":"/api/v1/runs/"+id}


@app.get("/api/v1/runs/{job_id}")
def result(job_id:str,authorization:str|None=Header(default=None)):
    require_auth(authorization)
    with LOCK:
        if job_id not in JOBS:raise HTTPException(404,"Job not found or expired")
        return dict(JOBS[job_id])
