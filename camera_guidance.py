from pathlib import Path

import cv2
from ultralytics import YOLO

try:
    from speech import speak
except ImportError:
    # Allows this file to be run directly from the nested project folder too.
    from ..speech import speak

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR if (BASE_DIR / "best.pt").exists() else BASE_DIR / "elevator_button_project"
MODEL_PATH = PROJECT_DIR / "best.pt"
SUPPORTED_FLOORS = {"1", "2", "3", "4"}


def get_floor() -> str:
    print("Supported floors: 1, 2, 3, 4")
    while True:
        floor = input("Enter required floor: ").strip()
        if floor in SUPPORTED_FLOORS:
            return floor
        message = "Invalid floor. Please enter 1, 2, 3, or 4."
        print(message)
        try:
            speak(message)
        except Exception as error:
            print(f"Voice output unavailable: {error}")


def main() -> None:
    print("=" * 50)
    print("AI-BASED ELEVATOR BUTTON RECOGNITION SYSTEM")
    print("=" * 50)
    print("YOLO Button Detection + Voice Guidance\n")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}\n"
            "Place best.pt inside elevator_button_project/."
        )

    floor = get_floor()
    target_button = f"btn_{floor}"
    model = YOLO(str(MODEL_PATH))

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Camera could not be opened. Check camera permissions or connection.")
        return

    print("Camera started. Press Q to quit.")
    last_guidance = ""
    candidate_guidance = ""
    candidate_count = 0
    stable_frames = 10

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                print("Could not read a camera frame.")
                break

            results = model(frame, conf=0.50, verbose=False)
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
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name} {confidence:.2f}",
                            (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)

                image_center = frame.shape[1] // 2
                cv2.line(frame, (image_center, 0),
                         (image_center, frame.shape[0]), (255, 255, 0), 2)

                if center_x < image_center - 100:
                    guidance = "MOVE LEFT"
                elif center_x > image_center + 100:
                    guidance = "MOVE RIGHT"
                else:
                    guidance = "BUTTON IN FRONT"

                if guidance == candidate_guidance:
                    candidate_count += 1
                else:
                    candidate_guidance = guidance
                    candidate_count = 1

                cv2.putText(frame, f"TARGET: FLOOR {floor}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(frame, f"DIRECTION: {guidance}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(frame, f"Stable: {candidate_count}/{stable_frames}",
                            (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2)

                if candidate_count >= stable_frames and guidance != last_guidance:
                    if guidance == "MOVE LEFT":
                        message = f"Floor {floor} button detected. Move left."
                    elif guidance == "MOVE RIGHT":
                        message = f"Floor {floor} button detected. Move right."
                    else:
                        message = f"Floor {floor} button detected. Button is in front of you."
                    print("Voice guidance:", message)
                    try:
                        speak(message)
                    except Exception as error:
                        print(f"Voice output unavailable: {error}")
                    last_guidance = guidance
                break

            if not found:
                cv2.putText(frame, f"SEARCHING FOR FLOOR {floor}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                candidate_guidance = ""
                candidate_count = 0

            cv2.imshow("AI Elevator Button Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    print("Camera stopped.\nProgram finished.")


if __name__ == "__main__":
    main()
