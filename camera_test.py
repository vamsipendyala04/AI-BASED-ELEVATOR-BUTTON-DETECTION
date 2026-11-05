import cv2


def main() -> None:
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Camera not working or access was denied.")
        return
    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                print("Could not read from camera.")
                break
            cv2.imshow("Camera Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
