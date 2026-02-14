# 🖐️ Hand Gesture Control System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

[Read in Ukrainian UA](README.uk.md)

Welcome to the **Hand Gesture Control System**! 🖱️👋
This project empowers you to control your computer mouse and execute commands using simple hand gestures. Powered by **Computer Vision** and **Deep Learning**, it offers a touch-free interface experience.

Built with powerful technologies:
-   **MediaPipe** for ultra-fast hand tracking.
-   **LSTM (Long Short-Term Memory)** neural networks for accurate gesture recognition.
-   **OpenCV** for image processing.

---

## 📑 Table of Contents

-   [✨ Features](#-features)
-   [📂 Project Structure](#-project-structure)
-   [🚀 Installation](#-installation)
    -   [Prerequisites](#prerequisites)
    -   [Setup Steps](#setup-steps)
-   [🕹️ Usage](#️-usage)
    -   [1. Data Collection](#1-data-collection-optional)
    -   [2. Model Training](#2-model-training)
    -   [3. Running the System](#3-running-the-system)
-   [⚙️ Configuration](#️-configuration)
-   [❓ Troubleshooting](#-troubleshooting)

---

## ✨ Features

-   **🖐️ Real-time Tracking**: Instantly detects and tracks hand landmarks with high precision.
-   **🧠 Smart Recognition**: Uses an LSTM model to understand dynamic gestures (motion sequences), not just static poses.
-   **🖱️ Mouse Control**: Move the cursor, click, drag, and scroll using natural hand movements.
-   **⚡ Low Latency**: Optimized for smooth performance on standard CPUs.
-   **🎨 Custom Gestures**: Comes with pre-train gestures, but you can easily add your own!
    -   `static` 🏋️ (No action)
    -   `swipe_right` ➡️ (Next window/tab)
    -   `swipe_left` ⬅️ (Previous window/tab)
    -   `ok` 👌 (Confirmation/Click)
    -   `stop` 🤚 (Pause/Hold)
    -   `browser` ✌️ (Open Browser)
    -   `fist_left` ✊ (Grab/Drag)

---

## 📂 Project Structure

Here's how the project is organized:

```text
HandGestureControl/
├── 📁 models/             # Contains trained model files (.pth)
├── 📁 src/                # Main source code directory
│   ├── 📄 config.py       # Central configuration (gestures, paths, params)
│   ├── 📄 main.py         # 🚀 Main entry point to run the controller
│   ├── 📄 model.py        # 🧠 Neural Network architecture (LSTM)
│   ├── 📄 train.py        # 🏋️ Script to train the model
│   ├── 📄 smart_collector.py # 📷 Tool for recording new gestures
│   ├── 📄 mouse_controller.py # 🖱️ Logic for mouse interaction
│   └── 📄 utils.py        # 🛠️ Helper functions
├── 📁 data/               # Raw training data (sequences of numpy arrays)
├── 📁 diagram/            # 📊 Training performance graphs
├── 📄 requirements.txt    # Python dependencies
└── 📄 README.md           # This documentation
```

---

## 🚀 Installation

### Prerequisites

-   **Python 3.8** or higher installed.
-   A working **Webcam**.
-   **Git** (optional, for cloning).

### Setup Steps

1.  **Clone the Repository**
    Open your terminal or command prompt and run:
    ```bash
    git clone <repository_url>
    cd HandGestureControl
    ```

2.  **Create a Virtual Environment** (Recommended)
    It's best to keep dependencies isolated.

    *   **Windows:**
        ```bash
        python -m venv venv
        venv\Scripts\activate
        ```
    *   **macOS / Linux:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Install Dependencies**
    Install all required libraries using `pip`:
    ```bash
    pip install -r requirements.txt
    ```
    > **Note:** Major libraries include `opencv-python`, `mediapipe`, `torch`, `pyautogui`, `scikit-learn`.

---

## 🕹️ Usage

### 1. Data Collection (Optional)
*Skip this if you just want to use the pre-trained model.*
To train the system on **your** specific hand movements:
```bash
python src/smart_collector.py
```
-   Follow the on-screen prompts.
-   Press `n` to switch gestures.
-   Press `Space` to record a sequence.
-   Data is saved to `data/<gesture_name>/`.

### 2. Model Training
After collecting data (or if you added new gestures), retrain the brain:
```bash
python src/train.py
```
-   The script automatically loads data, trains the LSTM, and validates accuracy.
-   The best model is saved as `models/gesture_lstm_best.pth`.
-   Check `diagram/` for accuracy/loss graphs! 📈

### 3. Running the System
Ready to control your mouse?
```bash
python src/main.py
```
-   **Controls**:
    -   **Move Mouse**: Point with your index finger.
    -   **Left Click**: Pinch Index + Thumb.
    -   **Right Click**: Pinch Middle + Thumb.
    -   **Scroll**: Raise Index + Middle fingers together.
-   **Exit**: Press `q` to quit the application.

---

## ⚙️ Configuration

Tweak the system in `src/config.py`:

| Variable | Description | Default |
| :--- | :--- | :--- |
| `GESTURES` | List of gesture names to recognize | `['static', 'swipe_right', ...]` |
| `SEQ_LENGTH` | Number of frames per gesture sequence | `30` |
| `THRESHOLD` | Confidence required to trigger action | `0.92` (92%) |
| `EPOCHS` | Training iterations | `150` |

---

## ❓ Troubleshooting

**Q: The system doesn't detect my hands.**
> **A:** Ensure lighting is good. Avoid backlighting. Keep your hands within the camera frame.

**Q: Mouse movement is jittery.**
> **A:** Increase the `smoothing` parameter in `mouse_controller.py` or adjust lighting.

**Q: Error: `Module not found`.**
> **A:** Make sure you activated your virtual environment and ran `pip install -r requirements.txt`.

**Q: Gestures are mixed up.**
> **A:** Try retraining the model with your own data using `smart_collector.py`. Everyone's hands move differently!

---

*Created with ❤️ for seamless Human-Computer Interaction.*
