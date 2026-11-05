from pathlib import Path

import cv2
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR / "elevator_button_project"
MODEL_PATH = PROJECT_DIR / "best.pt"
IMAGE_PATH = PROJECT_DIR / "test_images" / "20260227_174841_jpg.rf.326e98510d45c5510c7272533e7b0843.jpg"
SUPPORTED_FLOORS = {"1", "2", "3", "4"}


def main() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")
    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(f"Test image not found: {IMAGE_PATH}")

    while True:
        floor = input("Enter required floor (1, 2, 3, or 4): ").strip()
        if floor in SUPPORTED_FLOORS:
            break
        print("Invalid floor. Please enter 1, 2, 3, or 4.")

    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise RuntimeError(f"Could not read test image: {IMAGE_PATH}")

    model = YOLO(str(MODEL_PATH))
    target_button = f"btn_{floor}"
    results = model(image, conf=0.25, verbose=False)
    found = False

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[class_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        if class_name == target_button:
            print(f"Floor {floor} button found!")
            print(f"Confidence: {confidence:.2f}")
            print(f"Position: ({center_x}, {center_y})")
            found = True

    if not found:
        print(f"Floor {floor} button was not detected.")


if __name__ == "__main__":
    main()
