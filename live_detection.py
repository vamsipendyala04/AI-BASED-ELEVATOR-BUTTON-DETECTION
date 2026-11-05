from pathlib import Path

import cv2
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "elevator_button_project" / "best.pt"


def main() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open camera")
        return

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                print("Error: Could not read camera")
                break
            results = model(frame, conf=0.25, verbose=False)
            cv2.imshow("Elevator Button Detection", results[0].plot())
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
