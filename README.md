# ASL Game Tutor Project

## 📘 Project Overview
This project is an interactive **American Sign Language (ASL) Tutor Game** designed to help users learn and practice ASL alphabets using real-time feedback.

Unlike traditional static classifiers, this application uses an **Intelligent Agent** approach. It tracks the user's hand, classifies the gesture in real-time, and adapts the gameplay by suggesting words based on the user's performance history (Rationality).

## 🛠️ Tech Stack & Key Concepts
This project fulfills the "Integrative AI" requirement by combining two distinct AI paradigms:

1.  **Perception (Computer Vision & Deep Learning):**
    *   **Feature Extraction:** Uses **MediaPipe Hands** to extract 21 skeletal landmarks $(x, y)$ from the video feed. This makes the system robust to lighting changes and background noise compared to raw image classification.
    *   **Classification:** A custom **Multi-Layer Perceptron (MLP)** (Neural Network) trained on the feature set to classify 29 classes (A-Z, Space, Delete, Nothing).

2.  **Rationality (Rule-Based Agent):**
    *   **Tutor Logic:** A Rule-Based System that tracks the user's error rate for each letter.
    *   **Adaptive Curriculum:** The agent prioritizes words containing letters the user struggles with (e.g., if 'A' has a high error rate, the agent selects words like "APPLE").

## 📂 Project Structure
*   **`src/`**: Source code directory.
    *   `main.py`: The entry point for the game loop (Webcam UI + Agent integration).
    *   `tutor_logic.py`: The "Brain" of the agent (Mistake tracking & Word selection).
    *   `train_mlp.py`: Script to train the Neural Network.
    *   `create_dataset.py`: Feature engineering script (Image -> Landmarks CSV).
    *   `words.txt`: Dictionary of words for the game.
*   **`data/`**: Dataset directory.
    *   `processed/`: Contains `data.csv` (Landmark features) and `label_map.npy`.
*   **`models/`**: Saved models.
    *   `mlp_model.h5`: The trained Keras model used for inference.
*   **`requirements.txt`**: List of dependencies.

---

## 🚀 Installation & Setup Guide

### 1. Prerequisites
*   **Python 3.10 or 3.11** (Recommended).
*   A working webcam.

### 2. Set Up Virtual Environment (Highly Recommended)
To prevent dependency conflicts, please create a virtual environment:

**Windows:**
```bash
# Open terminal in project folder
python -m venv venv
.\venv\Scripts\activate

### Mac/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```
*(You should see `(venv)` at the start of your terminal line).*

### 3. Install Dependencies

Run the following command to install TensorFlow, MediaPipe, and OpenCV:

```bash
pip install -r requirements.txt
```

> **Note:** The requirements file includes `numpy<2` to prevent conflicts with TensorFlow.

---

## 🎮 How to Run

Navigate to the source folder:

```bash
cd src
```

Start the game:

```bash
python main.py
```

### Gameplay Instructions

*   **Right Hand Only:** Please use your **right hand** for signing (the model is trained on right-hand data).
*   **Mirroring:** The camera feed is mirrored horizontally for intuitive interaction.
*   **Goal:** Spell the target word displayed at the top.
*   **Feedback:**
    *   **Green Box:** Indicates the hand is detected.
    *   **Prediction:** Shows the letter the AI thinks you are signing.
    *   **Hold to Confirm:** Hold the correct sign for **~0.5 seconds** to register the letter.
*   **Controls:**
    *   **Q:** Quit the game.

---

## 🔧 Troubleshooting

1.  **"ModuleNotFoundError" or Import Errors**
    *   Ensure your virtual environment is active (`(venv)` is visible).
    *   Ensure you installed requirements: `pip install -r requirements.txt`.

2.  **TensorFlow Installation Fails (Windows)**
    *   If you get a "Long Path" error on Windows, try moving the project folder closer to the root drive (e.g., `C:\ASL_Project`).

3.  **Hand Not Detected / Jittery Box**
    *   Ensure your hand is well-lit.
    *   The system uses MediaPipe landmarks; ensure your full hand (palm and fingers) is visible to the camera.

---

## 📜 Requirements.txt Reference

(Ensure your `requirements.txt` contains exactly this):

```text
opencv-python
mediapipe
pandas
matplotlib
scikit-learn
tensorflow
numpy<2
```

---

## 📚 References

*   **Dataset:** ASL Alphabet Dataset (Kaggle)
*   **Frameworks:** TensorFlow/Keras, MediaPipe Hands
