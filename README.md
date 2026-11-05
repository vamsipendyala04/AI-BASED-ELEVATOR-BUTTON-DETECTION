# AI-Based Elevator Button Recognition and Voice Guidance

A computer-vision prototype that uses a trained YOLO object-detection model and OpenCV to detect elevator-panel buttons. A user selects a floor (1–4), and the application looks for the corresponding button (`btn_1` to `btn_4`) in the camera feed and provides simple left/right/in-front voice guidance using text-to-speech.

> **Safety notice:** This is an academic prototype, not a certified accessibility or navigation device. Its directional guidance is based only on the button's horizontal position in the camera image. It does not measure physical distance, understand obstacles, or guarantee that a button can be safely reached. Do not rely on it as the sole means of navigation.

## Features

- Real-time elevator-button detection using YOLO and OpenCV.
- Floor selection for floors 1, 2, 3, and 4.
- Simple visual direction cues: **MOVE LEFT**, **MOVE RIGHT**, or **BUTTON IN FRONT**.
- Voice announcements using `pyttsx3`.
- Single-image detection and webcam test scripts.
- Bundled trained weights (`elevator_button_project/best.pt`) for inference.

## Repository structure

```text
nft_project/
├── camera_guidance.py          # Main webcam application with floor selection and voice guidance
├── live_detection.py           # Webcam detection for all supported button classes
├── button_detection.py         # Detect a selected floor button in the sample image
├── voice_guidance.py           # Sample-image detection with voice guidance
├── camera_test.py              # Check webcam access
├── speech.py                   # pyttsx3 text-to-speech helper
├── requirements.txt            # Python dependencies
├── .gitignore
├── README.md
└── elevator_button_project/
    ├── best.pt                 # Trained YOLO weights
    ├── data.yaml               # Dataset configuration/metadata
    └── test_images/
        └── ...                 # Bundled sample test image
```

## Requirements

- Python 3.10 or newer is recommended.
- A webcam for live detection.
- Windows, macOS, or Linux. OpenCV must be able to open a camera and display windows.
- A configured system text-to-speech voice for `pyttsx3` voice output.

The model runs with Ultralytics/PyTorch. On first installation, pip may download the appropriate PyTorch packages for your platform. CPU inference is supported; a compatible NVIDIA GPU is optional.

## Setup

Run these commands from the repository's root directory (the directory containing `camera_guidance.py`).

### Windows (PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks environment activation, use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the project

### 1. Start the main live application

```bash
python camera_guidance.py
```

Enter a floor number from `1` to `4` when prompted. Point the webcam at an elevator button panel. The application highlights the requested button when detected and announces the direction. Press **Q** in the camera window to quit.

### 2. Test the webcam

```bash
python camera_test.py
```

Press **Q** to close the camera window.

### 3. View all detections live

```bash
python live_detection.py
```

### 4. Test a selected floor using the bundled image

```bash
python button_detection.py
```

### 5. Test sample-image detection with voice guidance

```bash
python voice_guidance.py
```

The sample-image scripts use the image stored in `elevator_button_project/test_images/`.

## Model and dataset information

- The repository includes the trained weights at `elevator_button_project/best.pt`.
- The model's class names are listed in `elevator_button_project/data.yaml` (including `btn_1` through `btn_4`).
- The original training/validation/test image directories referenced by `data.yaml` are **not included** in this repository. The included weights are intended for inference; reproducing training requires access to the original dataset and its license/terms.
- The dataset metadata in `data.yaml` identifies the Roboflow dataset as **CC BY 4.0**. If you redistribute the dataset or its images separately, follow the dataset's attribution and license requirements. The dataset license does not automatically establish the license for this source code or trained weights.

## Troubleshooting

| Problem | What to check |
|---|---|
| `Trained model not found` | Confirm `elevator_button_project/best.pt` exists and that you launched the command from the repository root. |
| `No module named ultralytics` / `cv2` / `pyttsx3` | Activate the virtual environment and run `pip install -r requirements.txt`. |
| Camera does not open | Close other apps using the webcam, check OS camera permissions, and verify the webcam with `python camera_test.py`. |
| No voice is heard | Check that your operating system has a working text-to-speech voice configured. Detection can still run if speech output is unavailable. |
| No target button is detected | Improve lighting, keep the panel in view, and ensure the panel resembles the examples used to train the model. Detection performance depends on the training data and camera conditions. |

## Limitations and future work

- Current floor selection is limited to floors 1–4.
- Direction is estimated from the target button's horizontal location in the image; there is no depth/distance estimation or obstacle detection.
- Detection quality may vary with lighting, reflections, camera angle, button design, and distance.
- Useful future improvements include broader dataset coverage, quantitative evaluation (precision, recall, mAP), configurable camera selection, distance estimation, and more robust accessibility testing.

## Acknowledgements

The dataset configuration references a Roboflow Universe elevator-button dataset. See the dataset URL and license details in `elevator_button_project/data.yaml` before reusing its data.

## License

No source-code license is currently specified. Unless a license is added by the project owner, GitHub users should not assume they have permission to reuse, modify, or redistribute the source code or model weights. Add a repository license only after confirming the rights to license all included project materials.
