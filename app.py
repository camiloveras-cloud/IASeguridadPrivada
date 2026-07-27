"""
Restricted-zone intrusion detection demo.

Detects people in a video/webcam feed, lets the operator draw a restricted
zone with the mouse, and raises a visual + audible alert plus a timestamped
snapshot whenever a detected person's ground point enters that zone.

The model only performs person detection. There is no notion of intent,
identity, or classification of the person as a threat -- alerts are labeled
"INTRUSION" / "person detected in restricted zone" only.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
INCIDENTS_DIR = OUTPUTS_DIR / "incidents"
ASSETS_DIR = BASE_DIR / "assets"
DEFAULT_ALARM_PATH = ASSETS_DIR / "alarm.wav"
DEFAULT_MODEL_PATH = BASE_DIR / "yolov8n.pt"

PERSON_CLASS_ID = 0  # COCO class 0 == "person"
WINDOW_NAME = "Restricted Zone Monitor"


def ensure_output_dirs() -> None:
    INCIDENTS_DIR.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect people entering a restricted zone from a video file or webcam.",
    )
    parser.add_argument(
        "--source",
        default="0",
        help='Video file path, or a camera index such as "0" for the default webcam.',
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu"],
        help="Inference device. Only CPU is supported in this demo.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.45,
        help="Minimum detection confidence (0-1) to consider a box a person.",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=5.0,
        help="Seconds to wait between alerts/snapshots once an intrusion is detected.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Restart the video from the beginning when it reaches the end (ignored for webcams).",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to the YOLO model weights (auto-downloaded on first use if missing).",
    )
    parser.add_argument(
        "--alarm",
        default=str(DEFAULT_ALARM_PATH),
        help="Path to a .wav file played on intrusion. Demo continues without audio if missing.",
    )
    parser.add_argument(
        "--mute",
        action="store_true",
        help="Disable audio alerts entirely.",
    )
    return parser.parse_args()


def resolve_source(source: str):
    """Return an int camera index if source looks numeric, else the string path."""
    try:
        return int(source)
    except ValueError:
        return source


class AlarmPlayer:
    """Plays a .wav alarm on a background thread so it never blocks the video loop."""

    def __init__(self, alarm_path: Path, muted: bool):
        self.available = False
        self.muted = muted
        self.alarm_path = alarm_path

        if muted:
            return

        if not alarm_path.exists():
            print(
                f"[WARN] Alarm file not found at '{alarm_path}'. "
                "Continuing without audio alerts."
            )
            return

        self.available = True

    def play(self) -> None:
        if not self.available or self.muted:
            return
        threading.Thread(target=self._play_blocking, daemon=True).start()

    def _play_blocking(self) -> None:
        try:
            if sys.platform == "darwin":
                import subprocess

                subprocess.run(
                    ["afplay", str(self.alarm_path)],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            elif sys.platform.startswith("win"):
                import winsound

                winsound.PlaySound(str(self.alarm_path), winsound.SND_FILENAME)
            else:
                import subprocess

                for player in ("paplay", "aplay"):
                    result = subprocess.run(
                        [player, str(self.alarm_path)],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    if result.returncode == 0:
                        break
        except Exception as exc:  # pragma: no cover - best-effort audio playback
            print(f"[WARN] Could not play alarm sound: {exc}")


@dataclass
class ZoneEditor:
    """Handles mouse-driven creation of the restricted-zone polygon."""

    points: list[tuple[int, int]] = field(default_factory=list)
    confirmed: bool = False

    def on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        if self.confirmed:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.points.clear()

    def confirm(self) -> bool:
        if len(self.points) >= 3:
            self.confirmed = True
            return True
        return False

    def reset(self) -> None:
        self.points.clear()
        self.confirmed = False

    def polygon(self) -> np.ndarray | None:
        if len(self.points) < 3:
            return None
        return np.array(self.points, dtype=np.int32)


def point_in_or_on_polygon(point: tuple[float, float], polygon: np.ndarray) -> bool:
    """True if point is inside the polygon OR exactly on its border."""
    result = cv2.pointPolygonTest(polygon, point, False)
    return result >= 0


def draw_zone(frame: np.ndarray, points: list[tuple[int, int]], confirmed: bool) -> None:
    if not points:
        return
    color = (0, 200, 255) if not confirmed else (0, 0, 255)
    for point in points:
        cv2.circle(frame, point, 4, color, -1)
    if len(points) >= 2:
        closed = confirmed
        cv2.polylines(frame, [np.array(points, dtype=np.int32)], closed, color, 2)


def draw_hud(
    frame: np.ndarray,
    alert_active: bool,
    incident_count: int,
    zone_confirmed: bool,
) -> None:
    h, w = frame.shape[:2]
    status_text = "ALERTA" if alert_active else "NORMAL"
    status_color = (0, 0, 255) if alert_active else (0, 200, 0)

    cv2.rectangle(frame, (0, 0), (w, 40), (30, 30, 30), -1)
    cv2.putText(
        frame, f"Estado: {status_text}", (10, 27),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame, f"Incidentes: {incident_count}", (250, 27),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
    )
    if not zone_confirmed:
        cv2.putText(
            frame,
            "Clic izq: punto | Clic der: borrar | Enter: confirmar zona | R: reiniciar | Q/Esc: salir",
            (10, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA,
        )

    if alert_active:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)


def save_incident_snapshot(frame: np.ndarray) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    out_path = INCIDENTS_DIR / f"incident_{timestamp}.jpg"
    cv2.imwrite(str(out_path), frame)
    return out_path


def open_capture(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        kind = "webcam" if isinstance(source, int) else f"video file '{source}'"
        print(f"[ERROR] Could not open {kind}. Check the source and try again.")
        return None
    return cap


def main() -> int:
    args = parse_args()
    ensure_output_dirs()

    source = resolve_source(args.source)
    is_webcam = isinstance(source, int)

    if not is_webcam:
        video_path = Path(args.source)
        if not video_path.exists():
            print(f"[ERROR] Video file not found: '{video_path}'.")
            return 1
        source = str(video_path)

    cap = open_capture(source)
    if cap is None:
        return 1

    print(f"[INFO] Loading model '{args.model}' on device '{args.device}' ...")
    try:
        from ultralytics import YOLO

        model = YOLO(args.model)
    except Exception as exc:
        print(f"[ERROR] Could not load detection model: {exc}")
        cap.release()
        return 1

    alarm = AlarmPlayer(Path(args.alarm), muted=args.mute)

    zone = ZoneEditor()
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, zone.on_mouse)

    incident_count = 0
    last_alert_time = 0.0

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                if not is_webcam and args.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                print("[INFO] End of video reached.")
                break

            polygon = zone.polygon()
            alert_active = False

            if zone.confirmed and polygon is not None:
                results = model.predict(
                    frame,
                    device=args.device,
                    conf=args.confidence,
                    classes=[PERSON_CLASS_ID],
                    verbose=False,
                )
                boxes = results[0].boxes

                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    ground_point = ((x1 + x2) / 2, y2)

                    inside = point_in_or_on_polygon(ground_point, polygon)
                    box_color = (0, 0, 255) if inside else (0, 200, 0)

                    cv2.rectangle(
                        frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2
                    )
                    cv2.putText(
                        frame, f"{conf:.2f}", (int(x1), max(0, int(y1) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2, cv2.LINE_AA,
                    )
                    cv2.circle(
                        frame, (int(ground_point[0]), int(ground_point[1])), 4, box_color, -1
                    )

                    if inside:
                        alert_active = True

            trigger_incident = False
            if alert_active:
                now = time.monotonic()
                if now - last_alert_time >= args.cooldown:
                    last_alert_time = now
                    incident_count += 1
                    trigger_incident = True

            draw_zone(frame, zone.points, zone.confirmed)
            draw_hud(frame, alert_active, incident_count, zone.confirmed)

            if trigger_incident:
                alarm.play()
                saved_path = save_incident_snapshot(frame)
                print(
                    f"[ALERT] Person detected in restricted zone. "
                    f"Incident #{incident_count} saved to '{saved_path}'."
                )

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):  # Q or Escape
                break
            elif key == 13:  # Enter
                if not zone.confirm():
                    print("[WARN] Draw at least 3 points before confirming the zone.")
            elif key in (ord("r"), ord("R")):
                zone.reset()
                print("[INFO] Zone reset. Draw a new polygon.")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"[INFO] Session ended. Total incidents: {incident_count}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
