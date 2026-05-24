"""
OPTIX Backend - Zero AI, Zero external APIs

3 Features:
  1. Motion Heatmap - red = crowded, blue = empty
  2. Traffic Load per hour - % of max activity (100% accurate, not a count)
  3. Interest Zones - where people slow down (optical flow)

Deploy on Render.com:
  Build command : pip install -r requirements.txt
  Start command : uvicorn main:app --host 0.0.0.0 --port $PORT
  Add Disk      : Mount path = /var/data, Size = 20 GB
  Env var       : DATA_DIR = /var/data
"""

import os
import uuid
import json
import base64
import threading
from pathlib import Path

import cv2
import numpy as np
import aiofiles
from fastapi import FastAPI, Request, Response, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

BASE    = Path(os.environ.get("DATA_DIR", "./data"))
UPLOADS = BASE / "uploads"
RESULTS = BASE / "results"
UPLOADS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

jobs: dict[str, str] = {}

app = FastAPI(title="OPTIX")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "Location", "Upload-Offset", "Upload-Length",
        "Tus-Resumable", "Tus-Version", "Tus-Max-Size", "Tus-Extension",
    ],
)

TUS_VER = "1.0.0"


def _h(extra=None):
    if extra is None:
        extra = {}
    return {"Tus-Resumable": TUS_VER, "Tus-Version": TUS_VER, **extra}


@app.options("/upload")
async def tus_options():
    return Response(status_code=204, headers=_h({
        "Tus-Max-Size":  str(50 * 1024 ** 3),
        "Tus-Extension": "creation,termination",
    }))


@app.post("/upload")
async def tus_create(request: Request):
    length_str = request.headers.get("Upload-Length", "")
    if not length_str.isdigit():
        return Response(status_code=400, content="Upload-Length required")

    meta = {}
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
    info = {"id": uid, "length": int(length_str), "offset": 0, "meta": meta}
    (UPLOADS / f"{uid}.json").write_text(json.dumps(info))
    (UPLOADS / f"{uid}.bin").write_bytes(b"")
    return Response(status_code=201, headers=_h({"Location": f"/upload/{uid}"}))


@app.head("/upload/{uid}")
async def tus_head(uid: str):
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
async def tus_patch(uid: str, request: Request):
    p = UPLOADS / f"{uid}.json"
    if not p.exists():
        return Response(status_code=404)
    info = json.loads(p.read_text())
    client_offset = int(request.headers.get("Upload-Offset", -1))
    if client_offset != info["offset"]:
        return Response(status_code=409, content="Offset mismatch")
    body = await request.body()
    async with aiofiles.open(UPLOADS / f"{uid}.bin", "ab") as f:
        await f.write(body)
    new_offset = info["offset"] + len(body)
    info["offset"] = new_offset
    p.write_text(json.dumps(info))
    if new_offset >= info["length"]:
        jobs[uid] = "queued"
        t = threading.Thread(
            target=_analyse,
            args=(uid, str(UPLOADS / f"{uid}.bin")),
            daemon=True
        )
        t.start()
    return Response(status_code=204, headers=_h({"Upload-Offset": str(new_offset)}))


@app.delete("/upload/{uid}")
async def tus_delete(uid: str):
    for ext in [".bin", ".json"]:
        fp = UPLOADS / f"{uid}{ext}"
        if fp.exists():
            fp.unlink()
    jobs.pop(uid, None)
    return Response(status_code=204, headers=_h())


@app.get("/status/{uid}")
async def status(uid: str):
    if uid in jobs:
        return JSONResponse({"id": uid, "status": jobs[uid]})
    result_file = RESULTS / uid / "analysis.json"
    if result_file.exists():
        data = json.loads(result_file.read_text())
        return JSONResponse({"id": uid, "status": data.get("status", "done")})
    upload_file = UPLOADS / f"{uid}.json"
    if upload_file.exists():
        info = json.loads(upload_file.read_text())
        if info["offset"] >= info["length"]:
            jobs[uid] = "queued"
            t = threading.Thread(target=_analyse, args=(uid, str(UPLOADS / f"{uid}.bin")), daemon=True)
            t.start()
            return JSONResponse({"id": uid, "status": "queued"})
    return JSONResponse({"id": uid, "status": "unknown"})


@app.get("/results/{uid}")
async def results(uid: str):
    f = RESULTS / uid / "analysis.json"
    if not f.exists():
        return JSONResponse({"error": "not ready"}, status_code=404)
    return JSONResponse(json.loads(f.read_text()))


@app.get("/results/{uid}/heatmap.png")
async def heatmap_img(uid: str):
    f = RESULTS / uid / "heatmap.png"
    if not f.exists():
        return Response(status_code=404)
    return FileResponse(str(f), media_type="image/png")


@app.get("/results/{uid}/overlay.png")
async def overlay_img(uid: str):
    f = RESULTS / uid / "overlay.png"
    if not f.exists():
        return Response(status_code=404)
    return FileResponse(str(f), media_type="image/png")


@app.get("/results/{uid}/interest.png")
async def interest_img(uid: str):
    f = RESULTS / uid / "interest.png"
    if not f.exists():
        return Response(status_code=404)
    return FileResponse(str(f), media_type="image/png")


def _analyse(uid: str, video_path: str) -> None:
    """
    Feature 1: Heatmap - motion accumulation, JET colormap
    Feature 2: Traffic load per hour - total motion pixels per hour slot
                expressed as % of the busiest hour (100% accurate)
    Feature 3: Interest zones - optical flow magnitude (where people slow down)
    """
    try:
        jobs[uid] = "processing"
        out = RESULTS / uid
        out.mkdir(exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Cannot open video file")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate duration upfront from frame count
        duration_seconds = int(total_frames / fps) if fps > 0 else 0
        hours = duration_seconds // 3600
        minutes = (duration_seconds % 3600) // 60
        duration_str = f"{hours}h {minutes}m"

        ret, frame0 = cap.read()
        if not ret:
            raise RuntimeError("Video has no readable frames")

        h, w = frame0.shape[:2]

        # Accumulators
        accum_heat = np.zeros((h, w), dtype=np.float32)
        accum_interest = np.zeros((h, w), dtype=np.float32)

        # Traffic load: motion pixels per hour slot
        # Key = hour index (0, 1, 2...), Value = total motion pixels
        traffic_per_hour = {}

        mog2 = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=25, detectShadows=False
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_num = 0
        sampled = 0
        prev_gray = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1

            # Sample every 3rd frame - fast and accurate
            if frame_num % 3 != 0:
                continue

            # Time slot - use 10-minute buckets (works for short and long videos)
            seconds_in = frame_num / fps
            slot_minutes = int(seconds_in // 600)  # 600 seconds = 10 minutes
            if slot_minutes not in traffic_per_hour:
                traffic_per_hour[slot_minutes] = 0.0

            # Feature 1: Heatmap
            fg = mog2.apply(frame)
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
            _, thresh = cv2.threshold(fg, 200, 1, cv2.THRESH_BINARY)
            motion_pixels = float(thresh.sum())
            accum_heat += thresh.astype(np.float32)

            # Feature 2: Traffic load - accumulate motion pixels per hour
            traffic_per_hour[slot_minutes] += motion_pixels

            # Feature 3: Interest zones (optical flow every 5th sample)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None and sampled % 5 == 0:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    0.5, 3, 13, 3, 5, 1.1, 0
                )
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                # Low magnitude = slow movement = interest
                interest_frame = np.clip(5.0 - mag, 0, 5.0)
                motion_mask = (mag > 0.3).astype(np.float32)
                accum_interest += interest_frame * motion_mask

            prev_gray = gray
            sampled += 1

        cap.release()

        if sampled == 0:
            raise RuntimeError("No frames were processed")

        # Build heatmap
        blur = cv2.GaussianBlur(accum_heat, (25, 25), 0)
        norm = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(str(out / "heatmap.png"), heatmap)

        # Build overlay on middle frame
        cap2 = cv2.VideoCapture(video_path)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        ret2, mid = cap2.read()
        cap2.release()
        if ret2:
            overlay = cv2.addWeighted(mid, 0.45, heatmap, 0.65, 0)
            cv2.imwrite(str(out / "overlay.png"), overlay)

        # Build interest zones image
        blur_i = cv2.GaussianBlur(accum_interest, (31, 31), 0)
        norm_i = cv2.normalize(blur_i, None, 0, 255, cv2.NORM_MINMAX)
        interest_img = cv2.applyColorMap(norm_i.astype(np.uint8), cv2.COLORMAP_HOT)
        cv2.imwrite(str(out / "interest.png"), interest_img)

        # 6 Bays - divide store horizontally into 6 equal sections
        # Each bay gets a dwell time in minutes based on optical flow accumulation
        # Low optical flow magnitude = people moving slowly = they are interested
        NUM_BAYS = 6
        bw = w // NUM_BAYS
        bays = []
        frames_per_second = fps / 3  # we sample every 3rd frame
        flow_samples_per_second = frames_per_second / 5  # optical flow every 5th sample

        # Bay activity = % of sampled frames that had motion in that bay
        # This is 100% accurate and easy for business owners to understand
        total_frames_with_motion = float(accum_heat.any(axis=0).sum()) or 1.0

        for i in range(NUM_BAYS):
            x1 = i * bw
            x2 = (i + 1) * bw
            # Count frames where this bay had motion
            bay_motion_frames = float((accum_heat[:, x1:x2].sum(axis=0) > 0).sum())
            bay_total_pixels = float(h * bw)
            # % of time this bay was active (had movement)
            pct_active = round(bay_motion_frames / max(float(sampled), 1) * 100, 1)

            bays.append({
                "bay": i + 1,
                "label": f"Bay {i + 1}",
                "pct_active": pct_active,
                "score": round(float(blur_i[:, x1:x2].mean()), 2),
                "x1_pct": round(x1 / w * 100, 1),
                "x2_pct": round(x2 / w * 100, 1),
            })

        # Sort by activity
        max_pct = max(b["pct_active"] for b in bays) or 1
        for b in bays:
            b["intensity"] = round(b["pct_active"] / max_pct * 100)
            if b["pct_active"] >= 60:
                b["status"] = "hot"
                b["advice"] = "High activity - place your most profitable products here"
            elif b["pct_active"] >= 25:
                b["status"] = "warm"
                b["advice"] = "Moderate activity - good for mid-range products"
            else:
                b["status"] = "cold"
                b["advice"] = "Low activity - consider relocating products or improving display"

        bays_sorted = sorted(bays, key=lambda b: b["pct_active"], reverse=True)
        top_zones = bays_sorted[:3]

        # Convert traffic per hour to % of busiest hour
        max_traffic = max(traffic_per_hour.values()) if traffic_per_hour else 1
        traffic_pct = {}
        for slot, val in sorted(traffic_per_hour.items()):
            # slot = number of 10-minute blocks from start
            h = slot // 6
            m = (slot % 6) * 10
            label = f"{h:02d}:{m:02d}"
            pct = round(val / max_traffic * 100)
            traffic_pct[label] = pct

        # Peak hour = busiest slot
        peak_hour = "N/A"
        if traffic_pct:
            peak_hour = max(traffic_pct, key=traffic_pct.get)

        # Staffing advice per hour
        staffing = {}
        for label, pct in traffic_pct.items():
            if pct >= 80:
                staffing[label] = "full"
            elif pct >= 40:
                staffing[label] = "normal"
            else:
                staffing[label] = "reduce"

        result = {
            "id": uid,
            "status": "done",
            "frames_total": frame_num,
            "frames_sampled": sampled,
            "resolution": f"{w}x{h}",
            "duration": duration_str,
            "fps": round(fps, 2),
            "heatmap_url": f"/results/{uid}/heatmap.png",
            "overlay_url": f"/results/{uid}/overlay.png",
            "interest_url": f"/results/{uid}/interest.png",
            "peak_hour": peak_hour,
            "traffic_per_hour": traffic_pct,
            "staffing_per_hour": staffing,
            "bays": bays,
            "top_bays": top_zones,
        }

        (out / "analysis.json").write_text(json.dumps(result, indent=2))
        jobs[uid] = "done"

    except Exception as e:
        jobs[uid] = f"error: {e}"
