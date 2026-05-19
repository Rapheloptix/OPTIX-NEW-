"""
OPTIX Backend - SAM (Segment Anything Model) by Meta

3 Features:
  1. Presence Heatmap - where detected objects appear most (red = frequent, blue = rare)
  2. Traffic Load per hour - object count per hour as % of peak hour
  3. Interest Zones - where objects dwell longest (mask overlap between frames)

SAM ViT-B checkpoint (~375 MB) is downloaded automatically on first run.
Override path via env var: SAM_CHECKPOINT=/path/to/sam_vit_b_01ec64.pth

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
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
import aiofiles
from fastapi import FastAPI, Request, Response, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

BASE    = Path(os.environ.get("DATA_DIR", "./data"))
UPLOADS = BASE / "uploads"
RESULTS = BASE / "results"
UPLOADS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

SAM_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
SAM_CHECKPOINT     = Path(os.environ.get("SAM_CHECKPOINT", str(BASE / "sam_vit_b_01ec64.pth")))
SAM_MODEL_TYPE     = os.environ.get("SAM_MODEL_TYPE", "vit_b")
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"

jobs: dict[str, str] = {}

_mask_generator: SamAutomaticMaskGenerator | None = None
_sam_lock = threading.Lock()


def _ensure_checkpoint() -> None:
    if SAM_CHECKPOINT.exists():
        return
    SAM_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading SAM checkpoint to {SAM_CHECKPOINT} (~375 MB)…")
    urllib.request.urlretrieve(SAM_CHECKPOINT_URL, str(SAM_CHECKPOINT))
    print("SAM checkpoint ready.")


def _get_mask_generator() -> SamAutomaticMaskGenerator:
    global _mask_generator
    if _mask_generator is None:
        with _sam_lock:
            if _mask_generator is None:
                _ensure_checkpoint()
                sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=str(SAM_CHECKPOINT))
                sam.to(device=DEVICE)
                _mask_generator = SamAutomaticMaskGenerator(
                    sam,
                    points_per_side=16,
                    pred_iou_thresh=0.86,
                    stability_score_thresh=0.92,
                    min_mask_region_area=500,
                )
    return _mask_generator


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
    return JSONResponse({"id": uid, "status": jobs.get(uid, "unknown")})


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
    Feature 1: Heatmap  - accumulate SAM segmentation masks across sampled frames
    Feature 2: Traffic  - count SAM-detected objects per hour slot, as % of peak
    Feature 3: Interest - mask overlap between consecutive frames (dwelling zones)
    """
    try:
        jobs[uid] = "processing"
        out = RESULTS / uid
        out.mkdir(exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Cannot open video file")

        fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        duration_seconds = int(total_frames / fps) if fps > 0 else 0
        hours   = duration_seconds // 3600
        minutes = (duration_seconds % 3600) // 60
        duration_str = f"{hours}h {minutes}m"

        ret, frame0 = cap.read()
        if not ret:
            raise RuntimeError("Video has no readable frames")

        h, w = frame0.shape[:2]

        accum_heat     = np.zeros((h, w), dtype=np.float32)
        accum_interest = np.zeros((h, w), dtype=np.float32)
        traffic_per_hour: dict[int, float] = {}

        # Cap at ~100 SAM runs regardless of video length (SAM is expensive on CPU)
        SAMPLE_EVERY = max(1, total_frames // 100)

        mask_gen  = _get_mask_generator()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_num = 0
        sampled   = 0
        prev_mask = np.zeros((h, w), dtype=np.float32)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1

            if frame_num % SAMPLE_EVERY != 0:
                continue

            seconds_in = frame_num / fps
            hour_slot  = int(seconds_in // 3600)
            traffic_per_hour.setdefault(hour_slot, 0.0)

            # SAM expects RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            sam_masks = mask_gen.generate(image_rgb)

            # Combine all plausible-sized object masks into one presence map
            combined     = np.zeros((h, w), dtype=np.float32)
            object_count = 0
            for m in sam_masks:
                area = m["area"]
                # Exclude tiny noise and full-frame background masks
                if 500 <= area <= h * w * 0.30:
                    combined += m["segmentation"].astype(np.float32)
                    object_count += 1
            combined = np.clip(combined, 0, 1)

            # Feature 1: where objects appear
            accum_heat += combined

            # Feature 2: object count as traffic proxy
            traffic_per_hour[hour_slot] += object_count

            # Feature 3: overlap with previous frame = slow / dwelling areas
            if sampled > 0:
                accum_interest += combined * prev_mask

            prev_mask = combined
            sampled  += 1

        cap.release()

        if sampled == 0:
            raise RuntimeError("No frames were processed")

        # --- Heatmap ---
        blur = cv2.GaussianBlur(accum_heat, (25, 25), 0)
        norm = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(str(out / "heatmap.png"), heatmap)

        # --- Overlay on middle frame ---
        cap2 = cv2.VideoCapture(video_path)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        ret2, mid = cap2.read()
        cap2.release()
        if ret2:
            overlay = cv2.addWeighted(mid, 0.45, heatmap, 0.65, 0)
            cv2.imwrite(str(out / "overlay.png"), overlay)

        # --- Interest zones image ---
        blur_i = cv2.GaussianBlur(accum_interest, (31, 31), 0)
        norm_i = cv2.normalize(blur_i, None, 0, 255, cv2.NORM_MINMAX)
        interest_img = cv2.applyColorMap(norm_i.astype(np.uint8), cv2.COLORMAP_HOT)
        cv2.imwrite(str(out / "interest.png"), interest_img)

        # --- Zone grid (6 x 6) ---
        ROWS, COLS = 6, 6
        zh, zw = h // ROWS, w // COLS
        zones = []
        for r in range(ROWS):
            for c in range(COLS):
                y1, y2 = r * zh, (r + 1) * zh
                x1, x2 = c * zw, (c + 1) * zw
                score  = float(blur_i[y1:y2, x1:x2].mean())
                zones.append({
                    "id":    f"R{r+1}C{c+1}",
                    "score": round(score, 2),
                    "x1":    round(x1 / w * 100, 1),
                    "y1":    round(y1 / h * 100, 1),
                    "x2":    round(x2 / w * 100, 1),
                    "y2":    round(y2 / h * 100, 1),
                })
        zones.sort(key=lambda z: z["score"], reverse=True)
        max_s = zones[0]["score"] if zones else 1
        for z in zones:
            z["intensity"] = round(z["score"] / max_s * 100) if max_s > 0 else 0
        top_zones = [z for z in zones[:3] if z["score"] > 0]

        # --- Traffic per hour as % of busiest hour ---
        max_traffic = max(traffic_per_hour.values()) if traffic_per_hour else 1
        traffic_pct: dict[str, int] = {}
        for slot, val in sorted(traffic_per_hour.items()):
            label = f"{slot:02d}:00"
            traffic_pct[label] = round(val / max_traffic * 100)

        peak_hour = max(traffic_pct, key=traffic_pct.get) if traffic_pct else "N/A"

        staffing: dict[str, str] = {}
        for label, pct in traffic_pct.items():
            if pct >= 80:
                staffing[label] = "full"
            elif pct >= 40:
                staffing[label] = "normal"
            else:
                staffing[label] = "reduce"

        result = {
            "id":                uid,
            "status":            "done",
            "frames_total":      frame_num,
            "frames_sampled":    sampled,
            "resolution":        f"{w}x{h}",
            "duration":          duration_str,
            "fps":               round(fps, 2),
            "heatmap_url":       f"/results/{uid}/heatmap.png",
            "overlay_url":       f"/results/{uid}/overlay.png",
            "interest_url":      f"/results/{uid}/interest.png",
            "peak_hour":         peak_hour,
            "traffic_per_hour":  traffic_pct,
            "staffing_per_hour": staffing,
            "interest_zones":    top_zones,
            "all_zones":         zones,
        }

        (out / "analysis.json").write_text(json.dumps(result, indent=2))
        jobs[uid] = "done"

    except Exception as e:
        jobs[uid] = f"error: {e}"
