"""
Browser front-end for the restricted-zone intrusion demo.

Same detection pipeline as app.py (person detection, ground-point zone
check, cooldown, evidence snapshots), but streamed as MJPEG to a web page
instead of a native OpenCV window. Meant for projecting the live demo to an
audience from a browser tab instead of a desktop window.

Run it, then open http://127.0.0.1:5000 in a browser:
    python web_app.py --source "Test Videos/Pencuri.mp4" --device cpu --loop
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, request, send_from_directory

from app import (
    ASSETS_DIR,
    DEFAULT_MODEL_PATH,
    PERSON_CLASS_ID,
    ZoneEditor,
    draw_hud,
    draw_zone,
    ensure_output_dirs,
    open_capture,
    point_in_or_on_polygon,
    resolve_source,
    save_incident_snapshot,
)

app = Flask(__name__)

zone = ZoneEditor()
zone_lock = threading.Lock()

state_lock = threading.Lock()
incident_count = 0
last_alert_time = 0.0
alert_active = False
stream_error: str | None = None

frame_lock = threading.Lock()
latest_jpeg: bytes | None = None

stop_event = threading.Event()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Browser (MJPEG) front-end for the restricted-zone intrusion demo.",
    )
    parser.add_argument("--source", default="0", help='Video file path, or camera index like "0".')
    parser.add_argument("--device", default="cpu", choices=["cpu"], help="Inference device (CPU only).")
    parser.add_argument("--confidence", type=float, default=0.45, help="Minimum detection confidence (0-1).")
    parser.add_argument("--cooldown", type=float, default=5.0, help="Seconds between alerts/snapshots.")
    parser.add_argument("--loop", action="store_true", help="Restart the video at the end (ignored for webcams).")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Path to YOLO model weights.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the web server to.")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind the web server to.")
    return parser.parse_args()


def capture_loop(args: argparse.Namespace) -> None:
    global latest_jpeg, incident_count, last_alert_time, alert_active, stream_error

    source = resolve_source(args.source)
    is_webcam = isinstance(source, int)

    if not is_webcam:
        video_path = Path(args.source)
        if not video_path.exists():
            stream_error = f"Video file not found: '{video_path}'."
            print(f"[ERROR] {stream_error}")
            return
        source = str(video_path)

    cap = open_capture(source)
    if cap is None:
        stream_error = "Could not open the video source. Check --source and try again."
        return

    print(f"[INFO] Loading model '{args.model}' on device '{args.device}' ...")
    try:
        from ultralytics import YOLO

        model = YOLO(args.model)
    except Exception as exc:
        stream_error = f"Could not load detection model: {exc}"
        print(f"[ERROR] {stream_error}")
        cap.release()
        return

    ensure_output_dirs()

    while not stop_event.is_set():
        ret, frame = cap.read()

        if not ret:
            if not is_webcam and args.loop:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            print("[INFO] End of video reached.")
            break

        with zone_lock:
            confirmed = zone.confirmed
            polygon = zone.polygon()
            points = list(zone.points)

        frame_alert = False

        if confirmed and polygon is not None:
            results = model.predict(
                frame, device=args.device, conf=args.confidence,
                classes=[PERSON_CLASS_ID], verbose=False,
            )
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                ground_point = ((x1 + x2) / 2, y2)
                inside = point_in_or_on_polygon(ground_point, polygon)
                color = (0, 0, 255) if inside else (0, 200, 0)

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"{conf:.2f}", (int(x1), max(0, int(y1) - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
                cv2.circle(frame, (int(ground_point[0]), int(ground_point[1])), 4, color, -1)

                if inside:
                    frame_alert = True

        trigger_incident = False
        with state_lock:
            alert_active = frame_alert
            if frame_alert:
                now = time.monotonic()
                if now - last_alert_time >= args.cooldown:
                    last_alert_time = now
                    incident_count += 1
                    trigger_incident = True
            current_count = incident_count

        draw_zone(frame, points, confirmed)
        draw_hud(frame, frame_alert, current_count, confirmed)

        if trigger_incident:
            saved_path = save_incident_snapshot(frame)
            print(
                f"[ALERT] Person detected in restricted zone. "
                f"Incident #{current_count} saved to '{saved_path}'."
            )

        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with frame_lock:
                latest_jpeg = buffer.tobytes()

    cap.release()
    print("[INFO] Capture loop stopped.")


def mjpeg_generator():
    while True:
        with frame_lock:
            jpeg_bytes = latest_jpeg
        if jpeg_bytes is None:
            time.sleep(0.05)
            continue
        yield (
            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
        )
        time.sleep(1 / 20)


INDEX_HTML = """<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Restricted Zone Monitor</title>
<style>
  body { background:#111; color:#eee; font-family: -apple-system, sans-serif; margin:0; padding:20px; }
  h1 { font-size: 1.1rem; font-weight: 600; margin: 0 0 12px; }
  #wrap { max-width: 900px; margin: 0 auto; }
  #stream { width: 100%; border: 2px solid #333; border-radius: 4px; cursor: crosshair; display:block; }
  #controls { margin-top: 12px; display:flex; gap:8px; flex-wrap: wrap; align-items:center; }
  button { background:#2a2a2a; color:#eee; border:1px solid #444; border-radius:4px; padding:8px 14px; cursor:pointer; font-size:0.9rem; }
  button:hover { background:#3a3a3a; }
  #status { margin-left:auto; font-size:0.9rem; }
  .normal { color:#3ecf5e; } .alerta { color:#ff4b4b; font-weight:700; }
  #help { margin-top:10px; font-size:0.8rem; color:#999; }
  #error { color:#ff4b4b; margin-top:10px; }
</style>
</head>
<body>
<div id="wrap">
  <h1>Restricted Zone Monitor — Web</h1>
  <img id="stream" src="/video_feed">
  <div id="controls">
    <button id="btnConfirm">Confirmar zona (Enter)</button>
    <button id="btnReset">Reiniciar zona (R)</button>
    <span id="status">Estado: <span id="statusText" class="normal">NORMAL</span> | Incidentes: <span id="count">0</span></span>
  </div>
  <div id="help">
    Clic izquierdo: agregar punto a la zona &middot; Clic derecho: borrar puntos &middot;
    Enter: confirmar zona &middot; R: reiniciar zona
  </div>
  <div id="error"></div>
</div>
<audio id="alarmAudio" src="/assets/alarm.wav" preload="auto"></audio>
<script>
const img = document.getElementById('stream');
const statusText = document.getElementById('statusText');
const countEl = document.getElementById('count');
const errorEl = document.getElementById('error');
const alarmAudio = document.getElementById('alarmAudio');
let lastCount = 0;

function toFrameCoords(evt) {
  const rect = img.getBoundingClientRect();
  const scaleX = (img.naturalWidth || rect.width) / rect.width;
  const scaleY = (img.naturalHeight || rect.height) / rect.height;
  return {
    x: Math.round((evt.clientX - rect.left) * scaleX),
    y: Math.round((evt.clientY - rect.top) * scaleY),
  };
}

img.addEventListener('click', (evt) => {
  const p = toFrameCoords(evt);
  fetch('/api/zone/point', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(p),
  });
});

img.addEventListener('contextmenu', (evt) => {
  evt.preventDefault();
  fetch('/api/zone/clear', { method: 'POST' });
});

document.getElementById('btnConfirm').addEventListener('click', () => {
  fetch('/api/zone/confirm', { method: 'POST' });
});
document.getElementById('btnReset').addEventListener('click', () => {
  fetch('/api/zone/reset', { method: 'POST' });
});

window.addEventListener('keydown', (evt) => {
  if (evt.key === 'Enter') fetch('/api/zone/confirm', { method: 'POST' });
  if (evt.key === 'r' || evt.key === 'R') fetch('/api/zone/reset', { method: 'POST' });
});

async function pollStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    statusText.textContent = data.alert_active ? 'ALERTA' : 'NORMAL';
    statusText.className = data.alert_active ? 'alerta' : 'normal';
    countEl.textContent = data.incident_count;
    if (data.incident_count > lastCount) {
      alarmAudio.currentTime = 0;
      alarmAudio.play().catch(() => {});
    }
    lastCount = data.incident_count;
    errorEl.textContent = data.error ? ('Error: ' + data.error) : '';
  } catch (e) {
    // stream/server not ready yet; ignore and retry
  }
}
setInterval(pollStatus, 500);
pollStatus();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return INDEX_HTML


@app.route("/video_feed")
def video_feed():
    return Response(mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/assets/<path:filename>")
def assets(filename):
    return send_from_directory(ASSETS_DIR, filename)


@app.route("/api/zone/point", methods=["POST"])
def api_zone_point():
    data = request.get_json(force=True)
    x, y = int(data["x"]), int(data["y"])
    with zone_lock:
        if not zone.confirmed:
            zone.points.append((x, y))
    return jsonify({"ok": True, "points": len(zone.points)})


@app.route("/api/zone/clear", methods=["POST"])
def api_zone_clear():
    with zone_lock:
        zone.points.clear()
    return jsonify({"ok": True})


@app.route("/api/zone/confirm", methods=["POST"])
def api_zone_confirm():
    with zone_lock:
        ok = zone.confirm()
    return jsonify({"ok": ok})


@app.route("/api/zone/reset", methods=["POST"])
def api_zone_reset():
    with zone_lock:
        zone.reset()
    with state_lock:
        global incident_count
        incident_count = 0
    return jsonify({"ok": True})


@app.route("/api/status")
def api_status():
    with state_lock:
        data = {
            "alert_active": alert_active,
            "incident_count": incident_count,
            "error": stream_error,
        }
    with zone_lock:
        data["zone_confirmed"] = zone.confirmed
        data["zone_points"] = len(zone.points)
    return jsonify(data)


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    worker = threading.Thread(target=capture_loop, args=(args,), daemon=True)
    worker.start()

    try:
        print(f"[INFO] Open http://{args.host}:{args.port} in a browser.")
        app.run(host=args.host, port=args.port, threaded=True, debug=False)
    finally:
        stop_event.set()


if __name__ == "__main__":
    main()
