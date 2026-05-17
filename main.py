"""
OPTIX Backend — Zero AI, Zero external APIs
============================================

3 Features:
  1. Motion Heatmap        — red = crowded, blue = empty
  2. People Counter        — count per hour from video timestamp
  3. Interest Zones        — where people slow down (optical flow magnitude)

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
import math
from pathlib import Path
from collections import OrderedDict

import cv2
import numpy as np
import aiofiles
from fastapi import FastAPI, Request, Response, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

# ── Storage ──────────────────────────────────────────────────────────────────
BASE    = Path(os.environ.get("DATA_DIR", "./data"))
UPLOADS = BASE / "uploads"
RESULTS = BASE / "results"
UPLOADS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

# ── Job state ─────────────────────────────────────────────────────────────────
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

def _h(extra: dict = {}) -> dict:
    return {"Tus-Resumable": TUS_VER, "Tus-Version": TUS_VER, **extra}


# ══════════════════════════════════════════════════════════════════════════════
#  TUS UPLOAD PROTOCOL
# ══════════════════════════════════════════════════════════════════════════════

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
async def tus_patch(uid: str, request: Request, bg: BackgroundTasks):
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
        bg.add_task(_analyse, uid, str(UPLOADS / f"{uid}.bin"))
    return Response(status_code=204, headers=_h({"Upload-Offset": str(new_offset)}))


@app.delete("/upload/{uid}")
async def tus_delete(uid: str):
    for ext in [".bin", ".json"]:
        fp = UPLOADS / f"{uid}{ext}"
        if fp.exists():
            fp.unlink()
    jobs.pop(uid, None)
    return Response(status_code=204, headers=_h())


# ══════════════════════════════════════════════════════════════════════════════
#  STATUS + RESULTS
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
#  CENTROID TRACKER — tracks blobs across frames, gives each a unique ID
#  No AI. Pure Euclidean distance math.
# ══════════════════════════════════════════════════════════════════════════════

class CentroidTracker:
    def __init__(self, max_disappeared=20):
        self.next_id    = 0
        self.objects    = OrderedDict()   # id → centroid
        self.disappeared = OrderedDict()  # id → frames since last seen
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_id]    = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, oid):
        del self.objects[oid]
        del self.disappeared[oid]

    def update(self, detections):
        """
        detections: list of (cx, cy) centroids from current frame.
        Returns dict of {id: (cx, cy)} for all currently tracked objects.
        """
        if len(detections) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.objects

        if len(self.objects) == 0:
            for d in detections:
                self.register(d)
        else:
            obj_ids       = list(self.objects.keys())
            obj_centroids = list(self.objects.values())

            # Euclidean distance matrix: existing objects vs new detections
            D = np.zeros((len(obj_centroids), len(detections)))
            for i, oc in enumerate(obj_centroids):
                for j, dc in enumerate(detections):
                    D[i, j] = math.sqrt((oc[0]-dc[0])**2 + (oc[1]-dc[1])**2)

            # Greedy match: row = existing, col = detection (smallest distance first)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for r, c in zip(rows, cols):
                if r in used_rows or c in used_cols:
                    continue
                if D[r, c] > 80:   # too far — not the same object
                    continue
                oid = obj_ids[r]
                self.objects[oid]    = detections[c]
                self.disappeared[oid] = 0
                used_rows.add(r)
                used_cols.add(c)

            # Unmatched existing objects — increment disappeared
            for r in range(len(obj_centroids)):
                if r not in used_rows:
                    oid = obj_ids[r]
                    self.disappeared[oid] += 1
                    if self.disappeared[oid] > self.max_disappeared:
                        self.deregister(oid)

            # New detections not matched — register as new
            for c in range(len(detections)):
                if c not in used_cols:
                    self.register(detections[c])

        return self.objects


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ANALYSIS — runs in background thread (plain def = auto-threaded by FastAPI)
# ══════════════════════════════════════════════════════════════════════════════

def _analyse(uid: str, video_path: str) -> None:
    """
    Processes the uploaded video and produces:

    1. heatmap.png      — motion accumulation map (red=crowded, blue=empty)
    2. overlay.png      — heatmap blended onto a real video frame
    3. interest.png     — interest zones from optical flow (where people slow down)
    4. analysis.json    — all numeric results:
         - people_per_hour : {"09:00": 12, "10:00": 34, ...}
         - peak_hour       : "14:00"
         - total_people    : 87
         - interest_zones  : top zones where people slow down
         - heatmap_url / overlay_url / interest_url
    """
    try:
        jobs[uid] = "processing"
        out = RESULTS / uid
        out.mkdir(exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Cannot open video file")

        fps         = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        ret, frame0 = cap.read()
        if not ret:
            raise RuntimeError("Video has no frames")

        h, w = frame0.shape[:2]

        # ── Accumulators ──────────────────────────────────────────────────────
        accum_heat     = np.zeros((h, w), dtype=np.float32)   # heatmap
        accum_interest = np.zeros((h, w), dtype=np.float32)   # interest zones
        people_per_hour: dict[str, int] = {}                   # "HH:00" → count
        counted_ids: set[int] = set()                          # track unique people

        # ── Tools ─────────────────────────────────────────────────────────────
        mog2 = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=25, detectShadows=False
        )
        tracker   = CentroidTracker(max_disappeared=15)
        kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Counting line = horizontal line at 50% of frame height
        count_line_y = h // 2
        # Track previous position of each ID to detect line crossing
        prev_positions: dict[int, int] = {}

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_num  = 0
        sampled    = 0
        prev_gray  = None  # for optical flow

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1

            # Process every 3rd frame — fast enough, accurate enough for 6h videos
            if frame_num % 3 != 0:
                continue

            # ── Current timestamp in video ────────────────────────────────────
            # frame_num / fps = seconds into the video
            seconds_in = frame_num / fps
            hour_slot  = int(seconds_in // 3600)         # 0, 1, 2, 3 ...
            hour_label = f"{hour_slot:02d}:00"           # "00:00", "01:00" ...
            if hour_label not in people_per_hour:
                people_per_hour[hour_label] = 0

            # ── Feature 1: Heatmap ────────────────────────────────────────────
            fg = mog2.apply(frame)
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  kernel)  # remove noise
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)  # fill holes
            _, thresh = cv2.threshold(fg, 200, 1, cv2.THRESH_BINARY)
            accum_heat += thresh.astype(np.float32)

            # ── Feature 2: People Counter ─────────────────────────────────────
            contours, _ = cv2.findContours(
                thresh.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            detections = []
            for c in contours:
                area = cv2.contourArea(c)
                if area < 400:    # ignore tiny blobs (noise)
                    continue
                if area > 50000:  # ignore huge blobs (multiple people merged — still count once)
                    pass
                M  = cv2.moments(c)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                detections.append((cx, cy))

            objects = tracker.update(detections)

            for oid, (cx, cy) in objects.items():
                prev_y = prev_positions.get(oid)
                if prev_y is not None:
                    # Count when centroid crosses the line (either direction)
                    crossed = (prev_y < count_line_y <= cy) or (prev_y > count_line_y >= cy)
                    if crossed and oid not in counted_ids:
                        counted_ids.add(oid)
                        people_per_hour[hour_label] += 1
                prev_positions[oid] = cy

            # ── Feature 3: Interest Zones (Optical Flow) ──────────────────────
            # Sample every 15th processed frame to save time (still accurate)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None and sampled % 5 == 0:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    pyr_scale=0.5, levels=3, winsize=13,
                    iterations=3, poly_n=5, poly_sigma=1.1, flags=0
                )
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                # LOW magnitude = slow movement = people are interested / stopping
                # Invert: interest = max_mag - mag  (so high interest = people stopped)
                interest_frame = np.clip(5.0 - mag, 0, 5.0)
                # Only count pixels where there IS motion (not pure background)
                motion_mask = (mag > 0.3).astype(np.float32)
                accum_interest += interest_frame * motion_mask

            prev_gray = gray
            sampled  += 1

        cap.release()

        if sampled == 0:
            raise RuntimeError("No frames were processed")

        # ── Build Heatmap ─────────────────────────────────────────────────────
        blur  = cv2.GaussianBlur(accum_heat, (25, 25), 0)
        norm  = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(str(out / "heatmap.png"), heatmap)

        # ── Build Overlay ─────────────────────────────────────────────────────
        cap2 = cv2.VideoCapture(video_path)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        ret2, mid = cap2.read()
        cap2.release()
        if ret2:
            overlay = cv2.addWeighted(mid, 0.45, heatmap, 0.65, 0)
            cv2.imwrite(str(out / "overlay.png"), overlay)

        # ── Build Interest Zone Image ─────────────────────────────────────────
        blur_i   = cv2.GaussianBlur(accum_interest, (31, 31), 0)
        norm_i   = cv2.normalize(blur_i, None, 0, 255, cv2.NORM_MINMAX)
        # Use HOT colormap: black=no interest, yellow=medium, white=maximum interest
        interest_img = cv2.applyColorMap(norm_i.astype(np.uint8), cv2.COLORMAP_HOT)
        cv2.imwrite(str(out / "interest.png"), interest_img)

        # ── Interest Zone Grid (6x6) ──────────────────────────────────────────
        ROWS, COLS = 6, 6
        zh, zw = h // ROWS, w // COLS
        zones = []
        for r in range(ROWS):
            for c in range(COLS):
                y1, y2 = r * zh, (r + 1) * zh
                x1, x2 = c * zw, (c + 1) * zw
                score = float(blur_i[y1:y2, x1:x2].mean())
                zones.append({
                    "id":    f"R{r+1}C{c+1}",
                    "score": round(score, 2),
                    "x1":   round(x1 / w * 100, 1),
                    "y1":   round(y1 / h * 100, 1),
                    "x2":   round(x2 / w * 100, 1),
                    "y2":   round(y2 / h * 100, 1),
                })
        zones.sort(key=lambda z: z["score"], reverse=True)
        max_s = zones[0]["score"] if zones else 1
        for z in zones:
            z["intensity"] = round(z["score"] / max_s * 100) if max_s > 0 else 0

        top_interest_zones = [z for z in zones[:3] if z["score"] > 0]

        # ── People Per Hour — sort by hour ────────────────────────────────────
        people_per_hour_sorted = dict(sorted(people_per_hour.items()))
        total_people = sum(people_per_hour_sorted.values())
        peak_hour    = max(people_per_hour_sorted, key=people_per_hour_sorted.get) \
                       if people_per_hour_sorted else "N/A"

        # ── Video duration ────────────────────────────────────────────────────
        duration_seconds = int(total_frames / fps)
        duration_str     = f"{duration_seconds // 3600}h {(duration_seconds % 3600) // 60}m"

        result = {
            "id":              uid,
            "status":          "done",
            "frames_total":    frame_num,
            "frames_sampled":  sampled,
            "resolution":      f"{w}x{h}",
            "duration":        duration_str,
            "fps":             round(fps, 2),

            # Feature 1 — Heatmap
            "heatmap_url":     f"/results/{uid}/heatmap.png",
            "overlay_url":     f"/results/{uid}/overlay.png",

            # Feature 2 — People Counter
            "total_people":    total_people,
            "peak_hour":       peak_hour,
            "people_per_hour": people_per_hour_sorted,

            # Feature 3 — Interest Zones
            "interest_url":    f"/results/{uid}/interest.png",
            "interest_zones":  top_interest_zones,
            "all_zones":       zones,
        }

        (out / "analysis.json").write_text(json.dumps(result, indent=2))
        jobs[uid] = "done"

    except Exception as e:
        jobs[uid] = f"error: {e}"
