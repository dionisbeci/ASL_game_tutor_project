import cv2
import mediapipe as mp
import os
import csv
import numpy as np

# --- Configuration ---
# Update this path to match your actual local structure
RAW_DATA_DIR = os.path.join("..", "data", "raw", "asl_alphabet_train", "asl_alphabet_train")
OUTPUT_DIR = os.path.join("..", "data", "processed")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "data.csv")
SAMPLES_PER_CLASS = 1000  # Limit to 1000 images per class for efficiency

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

def create_dataset():
    # 0. Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # 1. Prepare CSV Headers
    # 1 label + (21 landmarks * 2 coordinates) = 43 columns
    headers = ['label']
    for i in range(1, 22):
        headers.append(f'lm{i}_x')
        headers.append(f'lm{i}_y')

    if not os.path.exists(RAW_DATA_DIR):
        print(f"Error: Data directory not found at {RAW_DATA_DIR}")
        return

    # 2. Open CSV file for writing
    with open(OUTPUT_CSV, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        # 3. Loop through class folders
        classes = sorted(os.listdir(RAW_DATA_DIR))
        print(f"Found {len(classes)} classes. Starting feature extraction...")

        for label in classes:
            label_path = os.path.join(RAW_DATA_DIR, label)
            if not os.path.isdir(label_path):
                continue
            
            print(f"Processing Class: {label}...")
            images = os.listdir(label_path)[:SAMPLES_PER_CLASS]
            
            count = 0
            for img_name in images:
                img_path = os.path.join(label_path, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                # Convert BGR to RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        landmarks = []
                        
                        # Get Wrist coordinates (Landmark 0) for normalization
                        base_x = hand_landmarks.landmark[0].x
                        base_y = hand_landmarks.landmark[0].y

                        # Normalize: Subtract wrist coords from all landmarks
                        for lm in hand_landmarks.landmark:
                            landmarks.append(lm.x - base_x)
                            landmarks.append(lm.y - base_y)
                        
                        # Write to CSV: Label + 42 features
                        writer.writerow([label] + landmarks)
                        count += 1
            
            print(f"   Done. Extracted landmarks from {count} images for class {label}.")

    print(f"\nDataset creation complete! Saved as {OUTPUT_CSV}")

if __name__ == "__main__":
    create_dataset()
