import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import os
from tutor_logic import TutorAgent

# --- Configurations ---
# Paths relative to the 'src' folder
MODEL_PATH = os.path.join("..", "models", "mlp_model.h5")
LABEL_MAP_PATH = os.path.join("..", "models", "label_map_mlp.npy")
WORDS_FILE = "words.txt"

# --- Setup MediaPipe Hands ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# --- Load Model and Labels ---
print("Loading Landmark MLP Model...")
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model not found at {MODEL_PATH}")
    # Handle the case where script might be run from project root instead of src/
    MODEL_PATH = os.path.join("models", "mlp_model.h5")
    LABEL_MAP_PATH = os.path.join("models", "label_map_mlp.npy")
    WORDS_FILE = os.path.join("src", "words.txt")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    class_labels = np.load(LABEL_MAP_PATH, allow_pickle=True)
    print("Model and Labels loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- Initialize Tutor Agent ---
agent = TutorAgent(words_file_path=WORDS_FILE)

# --- Hint Image System ---
RAW_DATA_PATH = os.path.join("..", "data", "raw", "asl_alphabet_train", "asl_alphabet_train")
HINT_IMAGES = {}

def load_hint_images():
    """Caches one example image for each ASL letter."""
    print("Loading hint images...")
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Note: Raw data not found. Hints disabled.")
        return
    # Use labels from loaded mapping or common list
    for label in class_labels:
        label_dir = os.path.join(RAW_DATA_PATH, label)
        if os.path.exists(label_dir):
            images = [f for f in os.listdir(label_dir) if f.endswith(('.jpg', '.png'))]
            if images:
                img_path = os.path.join(label_dir, images[0])
                hint_img = cv2.imread(img_path)
                if hint_img is not None:
                    hint_img = cv2.resize(hint_img, (150, 150))
                    HINT_IMAGES[label] = hint_img

load_hint_images()

# --- Game State ---
current_word = agent.get_next_word()
current_letter_idx = 0
score = 0
debounce_counter = 0
MIN_THRESHOLD_FRAMES = 15 
CONFIDENCE_THRESHOLD = 0.8

cap = cv2.VideoCapture(0)

print("\n--- ASL Landmark Game Tutor Ready ---")
print("  Hold the sign for 15 frames to confirm.")
print("  'n': Skip Letter | 's': Skip Word | 'q': Quit")
print("--------------------------------------\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Mirror Effect: Flip horizontally
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    predicted_letter = "?"
    confidence = 0.0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on screen
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # --- Preprocessing: Extract and Normalize Landmarks ---
            landmarks = []
            
            # Wrist (Landmark 0) is the reference point for normalization
            wrist_x = hand_landmarks.landmark[0].x
            wrist_y = hand_landmarks.landmark[0].y
            
            for lm in hand_landmarks.landmark:
                # Relative Coordinates (Subtract wrist)
                landmarks.append(lm.x - wrist_x)
                landmarks.append(lm.y - wrist_y)
            
            # Convert to (1, 42) for prediction
            input_data = np.array([landmarks], dtype=np.float32)
            
            # Predict
            preds = model.predict(input_data, verbose=0)
            class_idx = np.argmax(preds)
            confidence = preds[0][class_idx]
            predicted_letter = class_labels[class_idx]

            # Visual feedback on prediction
            # Draw text near the wrist
            wrist_screen_x = int(wrist_x * w)
            wrist_screen_y = int(wrist_y * h)
            cv2.putText(frame, f"{predicted_letter} ({confidence*100:.0f}%)", 
                        (wrist_screen_x, wrist_screen_y - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # --- Game Logic ---
    target_letter = current_word[current_letter_idx].upper()
    
    if predicted_letter == target_letter and confidence > CONFIDENCE_THRESHOLD:
        debounce_counter += 1
    else:
        debounce_counter = 0
    
    # Check for Key Presses
    key = cv2.waitKey(1) & 0xFF
    
    # Advance Logic (Success or Skip)
    if debounce_counter >= MIN_THRESHOLD_FRAMES or key == ord('n'):
        if key == ord('n'):
            agent.update_performance(target_letter, False)
            print(f"Skipped letter: {target_letter}")
        else:
            print(f"Correct: {target_letter}!")
        
        debounce_counter = 0
        current_letter_idx += 1
        
        # Word Complete
        if current_letter_idx >= len(current_word):
            if key != ord('n'): # Only score if not skipped
                score += 1
                agent.update_performance(current_word[0], True)
            
            current_word = agent.get_next_word()
            current_letter_idx = 0
            print(f"Next Word: {current_word}")

    # Skip Word
    if key == ord('s'):
        for i in range(current_letter_idx, len(current_word)):
            agent.update_performance(current_word[i], False)
        current_word = agent.get_next_word()
        current_letter_idx = 0
        debounce_counter = 0
        print(f"Skipped Word. Next: {current_word}")

    # --- UI Overlay ---
    # Top Status Bar
    cv2.rectangle(frame, (0, 0), (w, 85), (0, 0, 0), -1)
    
    word_display = "".join([f"[{c}] " if i == current_letter_idx else f"{c} " 
                           for i, c in enumerate(current_word)])
    
    cv2.putText(frame, f"TARGET: {word_display}", (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    # Progress Bar (Debounce)
    if debounce_counter > 0:
        bar_w = int((debounce_counter / MIN_THRESHOLD_FRAMES) * 200)
        cv2.rectangle(frame, (20, 95), (20 + bar_w, 105), (0, 255, 0), -1)

    # Score and Labels
    cv2.putText(frame, f"SCORE: {score}", (w - 180, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, "[N] Skip Letter  [S] Skip Word", (20, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # --- Hint Overlay ---
    if target_letter in HINT_IMAGES:
        hint_img = HINT_IMAGES[target_letter]
        row, col, _ = hint_img.shape
        # Place hint in bottom-right
        frame[h-row-20:h-20, w-col-20:w-20] = hint_img
        cv2.putText(frame, f"Hint: {target_letter}", (w-col-20, h-row-25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(frame, (w-col-20, h-row-20), (w-20, h-20), (255, 255, 255), 1)

    cv2.imshow("ASL Landmark Game", frame)
    
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Final Score: {score}")
