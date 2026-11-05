from pathlib import Path

import cv2
from ultralytics import YOLO
from speech import speak

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
        print("Supported floors: 1, 2, 3, 4")
        floor = input("Enter required floor: ").strip()
        if floor in SUPPORTED_FLOORS:
            break
        message = "Invalid floor. Please enter 1, 2, 3, or 4."
        print(message)
        try:
            speak(message)
        except Exception as error:
            print(f"Voice output unavailable: {error}")

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

        if class_name != target_button:
            continue

        found = True
        print(f"Floor {floor} button detected")
        print(f"Confidence: {confidence:.2f}")
        print(f"Position: ({center_x}, {(y1 + y2) // 2})")
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, class_name, (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        image_center = image.shape[1] // 2
        if center_x < image_center - 100:
            message = f"Floor {floor} button detected. Move left."
        elif center_x > image_center + 100:
            message = f"Floor {floor} button detected. Move right."
        else:
            message = f"Floor {floor} button detected. Button is in front of you."
        print("Guidance:", message)
        try:
            speak(message)
        except Exception as error:
            print(f"Voice output unavailable: {error}")
        break

    if not found:
        message = f"Floor {floor} button was not detected."
        print(message)
        try:
            speak(message)
        except Exception as error:
            print(f"Voice output unavailable: {error}")
    else:
        output_path = BASE_DIR / "detected_result.jpg"
        cv2.imwrite(str(output_path), image)
        print(f"Result saved as: {output_path}")
        cv2.imshow("Elevator Button Detection", image)
        print("Press any key to close the image.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
