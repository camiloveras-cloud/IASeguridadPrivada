"""
Pre-flight check for the restricted-zone intrusion demo.

Run this before the live demo to catch problems early:
  - correct Python version
  - required packages importable
  - demo video present and readable
  - alarm audio present (optional, warns if missing)
  - YOLO model downloaded/cached
  - output folders created
  - a short real inference pass on the demo video
"""

from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
INCIDENTS_DIR = OUTPUTS_DIR / "incidents"
ASSETS_DIR = BASE_DIR / "assets"
DEFAULT_VIDEO = BASE_DIR / "Test Videos" / "Pencuri.mp4"
DEFAULT_ALARM = ASSETS_DIR / "alarm.wav"
DEFAULT_MODEL = BASE_DIR / "yolov8n.pt"

MIN_PY = (3, 10)
MAX_PY = (3, 12)  # exclusive upper bound; 3.10 and 3.11 are supported

CHECK_OK = "[OK]"
CHECK_WARN = "[WARN]"
CHECK_FAIL = "[FAIL]"


def check_python_version() -> bool:
    version = sys.version_info[:2]
    if MIN_PY <= version < MAX_PY:
        print(f"{CHECK_OK} Python {sys.version.split()[0]} is supported.")
        return True
    print(
        f"{CHECK_FAIL} Python {sys.version.split()[0]} detected. "
        f"This demo requires Python 3.10 or 3.11."
    )
    return False


def check_dependencies() -> bool:
    missing = []
    for module_name in ("cv2", "numpy", "ultralytics"):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)

    if missing:
        print(f"{CHECK_FAIL} Missing dependencies: {', '.join(missing)}.")
        print("       Run: pip install -r requirements.txt")
        return False

    print(f"{CHECK_OK} All required dependencies are importable (opencv, numpy, ultralytics).")
    return True


def check_video(video_path: Path) -> bool:
    import cv2

    if not video_path.exists():
        print(f"{CHECK_WARN} Demo video not found at '{video_path}'.")
        print("       You can still run the app with --source 0 for webcam, or --source <your_video>.")
        return True

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"{CHECK_FAIL} Demo video exists but could not be opened: '{video_path}'.")
        return False

    ret, _ = cap.read()
    cap.release()

    if not ret:
        print(f"{CHECK_FAIL} Demo video exists but no frame could be read: '{video_path}'.")
        return False

    print(f"{CHECK_OK} Demo video '{video_path.name}' opens and reads correctly.")
    return True


def check_audio(alarm_path: Path) -> bool:
    if not alarm_path.exists():
        print(f"{CHECK_WARN} Alarm file not found at '{alarm_path}'. Demo will run without audio alerts.")
        return True

    print(f"{CHECK_OK} Alarm file found at '{alarm_path}'.")
    return True


def prepare_model(model_path: Path) -> "object | None":
    try:
        from ultralytics import YOLO
    except ImportError:
        print(f"{CHECK_FAIL} ultralytics is not installed; cannot load the model.")
        return None

    try:
        print(f"[INFO] Loading/downloading model weights: '{model_path}' ...")
        model = YOLO(str(model_path))
        print(f"{CHECK_OK} Model ready ('{model_path.name}').")
        return model
    except Exception as exc:
        print(f"{CHECK_FAIL} Could not load or download the model: {exc}")
        return None


def create_output_dirs() -> bool:
    try:
        INCIDENTS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"{CHECK_OK} Output folder ready: '{INCIDENTS_DIR}'.")
        return True
    except OSError as exc:
        print(f"{CHECK_FAIL} Could not create output folders: {exc}")
        return False


def run_test_inference(model, video_path: Path, frames_to_test: int = 20) -> bool:
    if model is None:
        print(f"{CHECK_WARN} Skipping test inference: model not available.")
        return False

    if not video_path.exists():
        print(f"{CHECK_WARN} Skipping test inference: demo video not available.")
        return False

    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"{CHECK_FAIL} Could not open demo video for test inference.")
        return False

    # Start partway into the video: many demo clips open on an empty scene,
    # so sampling from the very first frames would under-report detections.
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if total_frames and total_frames > frames_to_test:
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames * 0.3)

    detections_seen = 0
    frames_read = 0

    for _ in range(frames_to_test):
        ret, frame = cap.read()
        if not ret:
            break
        frames_read += 1
        results = model.predict(frame, device="cpu", conf=0.45, classes=[0], verbose=False)
        detections_seen += len(results[0].boxes)

    cap.release()

    if frames_read == 0:
        print(f"{CHECK_FAIL} Test inference read zero frames.")
        return False

    print(
        f"{CHECK_OK} Test inference ran on {frames_read} frame(s), "
        f"{detections_seen} person detection(s) total."
    )
    return True


def main() -> int:
    print("=== Restricted Zone Demo: pre-flight check ===\n")

    ok = True
    ok &= check_python_version()
    ok &= check_dependencies()

    if not ok:
        print("\n[SUMMARY] Fix the errors above before running the demo.")
        return 1

    ok &= check_video(DEFAULT_VIDEO)
    ok &= check_audio(DEFAULT_ALARM)
    ok &= create_output_dirs()

    model = prepare_model(DEFAULT_MODEL)
    ok &= model is not None

    if model is not None:
        run_test_inference(model, DEFAULT_VIDEO)

    print()
    if ok:
        print("[SUMMARY] Everything looks ready. You can start the demo now:")
        print('  python app.py --source "Test Videos/Pencuri.mp4" --device cpu --loop')
    else:
        print("[SUMMARY] Some checks failed. Review the messages above before presenting.")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
