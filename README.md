# ASL Game Tutor Project

## Project Overview
This project is an interactive **American Sign Language (ASL) Tutor Game** that uses computer vision and machine learning to help users practice spelling words using ASL hand gestures. 

The application uses your computer's webcam to track your hand in real-time, recognizes the ASL letter you are signing, and guides you through spelling various words. It leverages **MediaPipe** for hand tracking and a custom-trained **Multi-Layer Perceptron (MLP)** neural network for accurate gesture classification.

## Key Concepts & Ideas
This project demonstrates several advanced concepts in AI and Software Engineering:

1.  **Computer Vision (CV):** 
    - Uses **MediaPipe Hands** to detect and track 21 distinct hand landmarks (joints and fingertips) in real-time.
    - Instead of processing the entire video frame (which is slow and sensitive to background noise), we focus only on the hand's geometry.

2.  **Feature Extraction:** 
    - **Raw Data:** The input is a raw image of a hand.
    - **Processed Data:** We convert this image into a set of **42 numerical numbers** (x and y coordinates for 21 landmarks).
    - **Normalization:** All coordinates are relative to the wrist (landmark 0) to ensure the model works regardless of where your hand is on the screen.

3.  **Supervised Learning (MLP):**
    - We use a **Multi-Layer Perceptron (MLP)**, a type of feedforward artificial neural network.
    - The model is trained on the extracted 42 features to classify them into 29 categories (A-Z, space, delete, nothing).

4.  **Game Logic:**
    - A "debounce" system ensures a gesture is held for a minimum number of frames before being accepted, preventing flickering inputs.
    - A state machine manages the flow of words, scoring, and user feedback.

## Project Structure & Paths
*   **`src/`**: Contains all the source code.
    *   `create_dataset.py`: Script to process raw images and extract landmark features into a CSV.
    *   `train_mlp.py`: Script to train the Neural Network using the CSV data.
    *   `main.py`: The main game application.
    *   `tutor_logic.py`: Helper class managing word selection and game state.
    *   `words.txt`: A list of words used in the game.
*   **`data/`**: Stores the datasets.
    *   `raw/`: Should contain the raw ASL image dataset (e.g., from Kaggle).
    *   `processed/`: Stores the generated `data.csv` used for training.
*   **`models/`**: Stores the trained resources.
    *   `mlp_model.h5`: The trained Tensorflow/Keras model.
    *   `label_map_mlp.npy`: The mapping between number IDs and letter labels.
*   **`requirements.txt`**: List of Python libraries required to run the project.

---

## 🚀 HOW TO RUN (Step-by-Step Guide)

Follow these steps exactly to set up and run the project on your computer.

### 1. Prerequisites
*   **Python:** Ensure you have Python installed (version 3.8 to 3.11 is recommended).
*   **Webcam:** A working webcam connected to your computer.

### 2. Installation
1.  **Download/Clone the Project:**
    Download this project folder to your computer.

2.  **Open Terminal:**
    Open your command prompt (cmd), PowerShell, or terminal in VS Code and navigate to the project folder:
    ```bash
    cd path/to/ASL_game_tutor_project
    ```

3.  **Install Dependencies:**
    Run the following command to install all necessary libraries:
    ```bash
    pip install -r requirements.txt
    ```

### 3. Data Setup (Only needed if retraining)
*If you already have the `models/mlp_model.h5` file, you can skip to Step 5.*

1.  **Download Data:** Download the "ASL Alphabet" dataset (e.g., from Kaggle).
2.  **Place Data:** Extract the images so they are in this exact structure:
    `data/raw/asl_alphabet_train/asl_alphabet_train/[A, B, C...]`
3.  **Generate Dataset:**
    Run the script to extract features from the images. This creates `data/processed/data.csv`.
    ```bash
    cd src
    python create_dataset.py
    ```

### 4. Training the Model (Only needed if retraining)
*If you already have the `models/mlp_model.h5` file, you can skip to Step 5.*

1.  Run the training script to teach the AI how to recognize gestures:
    ```bash
    cd src
    python train_mlp.py
    ```
    *This will create `models/mlp_model.h5` and `mlp_training_history.png`.*

### 5. Playing the Game
Now that everything is set up, you can run the main game!

1.  Navigate to the `src` folder (if not already there):
    ```bash
    cd src
    ```

2.  **Run the Game:**
    ```bash
    python main.py
    ```

### How to Play:
*   **Goal:** Spell the target word shown at the top of the screen.
*   **Action:** Make the ASL sign for the highlighted letter with your hand facing the camera.
*   **Confirm:** Hold the sign steady for about 0.5 seconds (the green bar will fill up) to confirm the letter.
*   **Controls:**
    *   `N` key: Skip the current letter.
    *   `S` key: Skip the current word.
    *   `Q` key: Quit the game.

**Troubleshooting:**
*   **Mirroring:** The camera is flipped horizontally (like a mirror) to make it easier to coordinate your movements.
*   **Lighting:** Ensure your hand is well-lit for the best recognition accuracy.
