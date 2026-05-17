"""
OPTIX Backend — Zero AI, Zero external APIs
============================================

Features:
  1. TUS resumable upload  — handles 5-15 GB videos, survives network drops
  2. OpenCV heatmap        — red = people crowded, blue = empty
  3. Bottleneck detection  — top 3 zones by motion density

Deploy on Render.com:
  1. Connect this repo
  2. Build command:  pip install -r requirements.txt
  3. Start command:  uvicorn main:app --host 0.0.0.0 --port $PORT
  4. Add Disk:       Mount path = /var/data,  Size = 20 GB
  5. Env var:        DATA_DIR = /var/data
"""

import os
import uuid
import json
import base64
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import aiofiles
from fastapi import FastAPI, Request, Response, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

# ── Storage ─────────────────────────────────────────────────────────────────
# Set DATA_DIR=/var/data on Render (the persistent disk mount path).
# Locally it falls back to ./data which is fine for testing.
BASE     = Path(os.environ.get("DATA_DIR", "./data"))
UPLOADS  = BASE / "uploads"
RESULTS  = BASE / "results"
UPLOADS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

# ── Job state ────────────────────────────────────────────────────────────────
# Simple dict — good enough for a single-server MVP
# key = upload_id
# value = "uploading" | "queued" | "processing" | "done" | "error:<msg>"
jobs: dict[str, str] = {}

# OpenCV is CPU-heavy. Run it in a thread so FastAPI stays responsive.
_pool = ThreadPoolExecutor(max_workers=2)

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="OPTIX")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # lock down to your Vercel URL in production
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "Location", "Upload-Offset", "Upload-Length",
        "Tus-Resumable", "Tus-Version", "Tus-Max-Size", "Tus-Extension",
    ],
)

TUS_VER = "1.0.0"

def _h(extra: dict = {}) -> dict:
    """Add required TUS headers to every response."""
    return {"Tus-Resumable": TUS_VER, "Tus-Version": TUS_VER, **extra}


# ═══════════════════════════════════════════════════════════════════════════
#  TUS UPLOAD PROTOCOL
#  Each chunk is a small PATCH request (5 MB default in tus-js-client).
#  Render's 100-minute timeout per request is never hit because each
#  individual chunk completes in seconds.
# ═══════════════════════════════════════════════════════════════════════════

@app.options("/upload")
async def tus_options():
    """Step 0 — browser asks what the server supports."""
    return Response(status_code=204, headers=_h({
        "Tus-Max-Size":   str(50 * 1024 ** 3),   # 50 GB hard cap
        "Tus-Extension":  "creation,termination",
    }))


@app.post("/upload")
async def tus_create(request: Request):
    """Step 1 — client creates an upload session, gets back a unique URL."""
    length_str = request.headers.get("Upload-Length", "")
    if not length_str.isdigit():
        return Response(status_code=400, content="Upload-Length header required")

    # Decode Upload-Metadata  (base64 key-value pairs separated by commas)
    meta: dict[str, str] = {}
    for pair in request.headers.get("Upload-Metadata", "").split(","):
        pair = pair.strip()
        if " " in pair:
            k, v = pair.split(" ", 1)
            try:
                meta[k] = base64.b64decode(v).decode()
            except Exception:
                meta[k] = v

    uid = str(uuid.uuid4())
    jobs[uid] = "uploading"

    # Persist upload metadata so HEAD works after a server restart
    info = {"id": uid, "length": int(length_str), "offset": 0, "meta": meta}
    (UPLOADS / f"{uid}.json").write_text(json.dumps(info))
    (UPLOADS / f"{uid}.bin").write_bytes(b"")   # empty file, will grow chunk by chunk

    return Response(status_code=201, headers=_h({"Location": f"/upload/{uid}"}))


@app.head("/upload/{uid}")
async def tus_head(uid: str):
    """Step 2 — client asks how many bytes you already have (for resume)."""
    p = UPLOADS / f"{uid}.json"
    if not p.exists():
        return Response(status_code=404)
    info = json.loads(p.read_text())
    return Response(status_code=200, headers=_h({
        "Upload-Offset": str(info["offset"]),
        "Upload-Length": str(info["length"]),
        "Cache-Control": "no-store",
    }))


@app.patch("/upload/{uid}")
async def tus_patch(uid: str, request: Request, bg: BackgroundTasks):
    """Step 3 — client sends one chunk. We append it. Repeat until done."""
    p = UPLOADS / f"{uid}.json"
    if not p.exists():
        return Response(status_code=404)

    info = json.loads(p.read_text())
    client_offset = int(request.headers.get("Upload-Offset", -1))

    if client_offset != info["offset"]:
        # Offset mismatch — client and server disagree. 409 tells tus-js-client
        # to do a HEAD request and re-sync. It handles this automatically.
        return Response(status_code=409, content="Offset mismatch")

    # Read and append the chunk
    body = await request.body()
    async with aiofiles.open(UPLOADS / f"{uid}.bin", "ab") as f:
        await f.write(body)

    new_offset = info["offset"] + len(body)
    info["offset"] = new_offset
    p.write_text(json.dumps(info))

    # All bytes received — trigger background OpenCV analysis
    if new_offset >= info["length"]:
        jobs[uid] = "queued"
        video_path = str(UPLOADS / f"{uid}.bin")
        # add_task with a plain def runs it in a threadpool automatically (FastAPI does this)
        bg.add_task(_analyse, uid, video_path)

    return Response(status_code=204, headers=_h({"Upload-Offset": str(new_offset)}))


@app.delete("/upload/{uid}")
async def tus_delete(uid: str):
    """Client cancelled the upload."""
    for ext in [".bin", ".json"]:
        fp = UPLOADS / f"{uid}{ext}"
        if fp.exists():
            fp.unlink()
    jobs.pop(uid, None)
    return Response(status_code=204, headers=_h())


# ═══════════════════════════════════════════════════════════════════════════
#  STATUS + RESULTS
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/status/{uid}")
async def status(uid: str):
    return JSONResponse({"id": uid, "status": jobs.get(uid, "unknown")})


@app.get("/results/{uid}")
async def results(uid: str):
    f = RESULTS / uid / "analysis.json"
    if not f.exists():
        return JSONResponse({"error": "not ready"}, status_code=404)
    return JSONResponse(json.loads(f.read_text()))


@app.get("/results/{uid}/heatmap.png")
async def heatmap_image(uid: str):
    f = RESULTS / uid / "heatmap.png"
    if not f.exists():
        return Response(status_code=404)
    return FileResponse(str(f), media_type="image/png")


# ═══════════════════════════════════════════════════════════════════════════
#  OPENCV ANALYSIS  —  pure math, zero AI, zero API calls
#
#  This runs in a background thread (FastAPI calls plain def functions
#  automatically in a threadpool, so the event loop stays unblocked).
# ═══════════════════════════════════════════════════════════════════════════

def _analyse(uid: str, video_path: str) -> None:
    """
    1. Open the video with OpenCV.
    2. MOG2 background subtraction — strips static background (walls, shelves).
       What's left = pixels that moved = people.
    3. Accumulate moving pixels across the whole video into one image.
       Bright spots = places people visited often.
    4. Apply JET colormap → heatmap.png  (red=crowded, blue=empty)
    5. Divide into 6×6 grid, score each cell, pick top 3 = bottlenecks.
    """
    try:
        jobs[uid] = "processing"
        out = RESULTS / uid
        out.mkdir(exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Cannot open video file")

        ret, frame0 = cap.read()
        if not ret:
            raise RuntimeError("Video has no readable frames")

        h, w = frame0.shape[:2]

        # Float accumulator — starts at zero, we add motion pixels each frame
        accum = np.zeros((h, w), dtype=np.float32)

        # MOG2: learns the static background, returns a mask of moving objects
        # history=500  → uses last 500 frames to model background
        # varThreshold=50  → sensitivity; lower = picks up more movement
        # detectShadows=False  → faster, we don't need shadow detection
        mog2 = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=False
        )

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        total   = 0
        sampled = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            total += 1

            # Sample every 5th frame — plenty accurate, 5× faster
            if total % 5 != 0:
                continue

            fg = mog2.apply(frame)

            # Only count strong motion (value > 200), discard noise
            _, thresh = cv2.threshold(fg, 200, 1, cv2.THRESH_BINARY)
            accum += thresh.astype(np.float32)
            sampled += 1

        cap.release()

        if sampled == 0:
            raise RuntimeError("No frames could be sampled from this video")

        # ── Heatmap ──────────────────────────────────────────────────────
        norm     = cv2.normalize(accum, None, 0, 255, cv2.NORM_MINMAX)
        heatmap  = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(str(out / "heatmap.png"), heatmap)

        # ── Bottleneck grid ───────────────────────────────────────────────
        ROWS, COLS = 6, 6
        zh = h // ROWS
        zw = w // COLS
        zones = []

        for r in range(ROWS):
            for c in range(COLS):
                y1, y2 = r * zh, (r + 1) * zh
                x1, x2 = c * zw, (c + 1) * zw
                score = float(accum[y1:y2, x1:x2].sum())
                zones.append({
                    "id":    f"R{r+1}C{c+1}",
                    "score": round(score, 1),
                    # percentage coords so the frontend can overlay on any screen size
                    "x1": round(x1 / w * 100, 1),
                    "y1": round(y1 / h * 100, 1),
                    "x2": round(x2 / w * 100, 1),
                    "y2": round(y2 / h * 100, 1),
                })

        zones.sort(key=lambda z: z["score"], reverse=True)
        max_score = zones[0]["score"] if zones else 1

        bottlenecks = [
            {**z, "intensity": round(z["score"] / max_score * 100)}
            for z in zones[:3]
            if z["score"] > 0
        ]

        result = {
            "id":             uid,
            "status":         "done",
            "frames_total":   total,
            "frames_sampled": sampled,
            "resolution":     f"{w}x{h}",
            "heatmap_url":    f"/results/{uid}/heatmap.png",
            "bottlenecks":    bottlenecks,
            "all_zones":      zones,
        }

        (out / "analysis.json").write_text(json.dumps(result, indent=2))
        jobs[uid] = "done"

    except Exception as e:
        jobs[uid] = f"error: {e}"
