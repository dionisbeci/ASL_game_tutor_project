import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import os
import sys
import traceback
import threading

# Add the current directory to sys.path to allow importing tutor_logic
sys.path.append(os.path.dirname(__file__))
from tutor_logic import TutorAgent

# --- Configuration ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "mlp_model.h5")
LABEL_MAP_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "label_map_mlp.npy")
RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "asl_alphabet_train", "asl_alphabet_train")

# --- Resource Loading (Cached) ---
@st.cache_resource
def load_resources():
    resources = {}
    
    # 1. Load Model & Map
    try:
        if os.path.exists(MODEL_PATH):
            resources['model'] = tf.keras.models.load_model(MODEL_PATH)
            resources['labels'] = np.load(LABEL_MAP_PATH, allow_pickle=True)
        else:
            return None
    except Exception:
        return None

    # 2. Load Hint Images
    hints = {}
    if os.path.exists(RAW_DATA_PATH) and 'labels' in resources:
        try:
            for label in resources['labels']:
                label_dir = os.path.join(RAW_DATA_PATH, label)
                if os.path.exists(label_dir):
                    images = [f for f in os.listdir(label_dir) if f.endswith(('.jpg', '.png'))]
                    if images:
                        img_path = os.path.join(label_dir, images[0])
                        hint_img = cv2.imread(img_path)
                        if hint_img is not None:
                            hint_img = cv2.resize(hint_img, (150, 150))
                            hints[label] = hint_img
        except Exception:
            pass
    
    resources['hints'] = hints
    return resources

# Load global resources once
resources = load_resources()
model = resources['model'] if resources else None
class_labels = resources['labels'] if resources else None
hint_images = resources['hints'] if resources else {}

# --- Initializes MediaPipe ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0, 
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# --- Video Processor with Integrated Game State ---
class ASLProcessor(VideoProcessorBase):
    def __init__(self):
        # Game State
        words_file = os.path.join(os.path.dirname(__file__), "words.txt")
        self.agent = TutorAgent(words_file_path=words_file)
        self.current_word = self.agent.get_next_word()
        self.word_idx = 0
        self.score = 0
        
        # Debounce
        self.debounce_counter = 0
        self.min_threshold_frames = 10 
        
        # Optimization: Frame Skipping
        self.frame_count = 0
        self.skip_rate = 3 # Process 1 in 3 frames (10fps inference @ 30fps)
        self.last_hand_landmarks = None
        self.last_prediction = "?"
        self.last_confidence = 0.0
        
        # Thread Lock just in case
        self.lock = threading.Lock()

    def skip_letter_action(self):
        with self.lock:
            if self.current_word:
                target_char = self.current_word[self.word_idx]
                self.agent.update_performance(target_char, False)
                
                self.word_idx += 1
                if self.word_idx >= len(self.current_word):
                    # Logic: Skipping last letter finishes word, no point
                    self.current_word = self.agent.get_next_word()
                    self.word_idx = 0

    def skip_word_action(self):
        with self.lock:
            if self.current_word:
                for i in range(self.word_idx, len(self.current_word)):
                    self.agent.update_performance(self.current_word[i], False)
                
                self.current_word = self.agent.get_next_word()
                self.word_idx = 0
    
    def reset_progress(self):
         with self.lock:
            self.score = 0
            # Reload agent to reset mistake counts
            words_file = os.path.join(os.path.dirname(__file__), "words.txt")
            self.agent = TutorAgent(words_file_path=words_file)
            self.current_word = self.agent.get_next_word()
            self.word_idx = 0

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            
            with self.lock:
                self.frame_count += 1
                
                # 1. Mirror
                img = cv2.flip(img, 1)
                h, w, _ = img.shape
                
                # 2. AI Processing (Skipping)
                if self.frame_count % self.skip_rate == 0:
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb_img)
                    
                    self.last_hand_landmarks = None
                    self.last_prediction = "?"
                    
                    if results.multi_hand_landmarks:
                        self.last_hand_landmarks = results.multi_hand_landmarks[0]
                        
                        # Features
                        landmarks = []
                        wrist_x = self.last_hand_landmarks.landmark[0].x
                        wrist_y = self.last_hand_landmarks.landmark[0].y
                        for lm in self.last_hand_landmarks.landmark:
                            landmarks.append(lm.x - wrist_x)
                            landmarks.append(lm.y - wrist_y)
                        
                        # Predict
                        if model is not None:
                            input_data = np.array([landmarks], dtype=np.float32)
                            preds = model.predict(input_data, verbose=0)
                            class_idx = np.argmax(preds)
                            self.last_confidence = preds[0][class_idx]
                            if class_labels is not None:
                                self.last_prediction = class_labels[class_idx]
                
                # 3. Game Logic
                target_word = self.current_word
                target_letter = target_word[self.word_idx].upper() if target_word else ""
                
                predicted_letter = self.last_prediction
                conf = self.last_confidence
                
                if predicted_letter == target_letter and conf > 0.8:
                    self.debounce_counter += 1
                else:
                    self.debounce_counter = 0
                
                if self.debounce_counter >= self.min_threshold_frames:
                    self.debounce_counter = 0
                    self.agent.update_performance(target_letter, True)
                    self.word_idx += 1
                    if self.word_idx >= len(target_word):
                        self.score += 1
                        self.current_word = self.agent.get_next_word()
                        self.word_idx = 0
                        
                # 4. Drawing (Use Cache)
                if self.last_hand_landmarks:
                    mp_draw.draw_landmarks(img, self.last_hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Top Bar
                cv2.rectangle(img, (0, 0), (w, 85), (0, 0, 0), -1)
                
                # Target Word
                word_display = "".join([f"[{c}] " if i == self.word_idx else f"{c} " 
                                       for i, c in enumerate(target_word)])
                cv2.putText(img, f"TARGET: {word_display}", (20, 35), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                
                # Score
                cv2.putText(img, f"SCORE: {self.score}", (w - 180, 35), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                            
                # Controls Hints text
                cv2.putText(img, "[N] Skip Letter  [S] Skip Word", (20, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
                
                # Debounce Bar
                if self.debounce_counter > 0:
                    bar_w = int((self.debounce_counter / self.min_threshold_frames) * 200)
                    cv2.rectangle(img, (20, 95), (20 + bar_w, 105), (0, 255, 0), -1)
                
                # Prediction Overlay
                if predicted_letter != "?":
                    color = (0, 255, 0) if predicted_letter == target_letter else (0, 0, 255)
                    # Position relative to wrist
                    if self.last_hand_landmarks:
                        wrist_x = int(self.last_hand_landmarks.landmark[0].x * w)
                        wrist_y = int(self.last_hand_landmarks.landmark[0].y * h)
                        # Fix out of bounds
                        wrist_x = max(20, min(w-50, wrist_x))
                        wrist_y = max(50, min(h-20, wrist_y))
                        
                        cv2.putText(img, f"{predicted_letter} ({int(conf*100)}%)", 
                                    (wrist_x, wrist_y - 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Hint Image
                if target_letter in hint_images:
                    hint = hint_images[target_letter]
                    h_img, w_img, _ = hint.shape
                    if h > h_img and w > w_img:
                        y_off = h - h_img - 20
                        x_off = w - w_img - 20
                        img[y_off:y_off+h_img, x_off:x_off+w_img] = hint
                        cv2.putText(img, f"Hint: {target_letter}", (x_off, y_off - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.rectangle(img, (x_off, y_off), (x_off+w_img, y_off+h_img), (255, 255, 255), 1)

            return av.VideoFrame.from_ndarray(img, format="bgr24")
            
        except Exception as e:
            print(f"Error: {e}")
            return frame

# --- Main Layout ---
st.set_page_config(layout="wide", page_title="ASL Game Tutor")
st.title("ASL Game Tutor")

col1, col2 = st.columns([2, 1])

with col1:
    ctx = webrtc_streamer(
        key="asl-tutor",
        video_processor_factory=ASLProcessor,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={
            "video": {
                "width": 640,
                "height": 480,
                "frameRate": 30
            }, 
            "audio": False
        },
        async_processing=True
    )

with col2:
    st.markdown("### Controls")
    c1, c2 = st.columns(2)
    
    with c1:
        if st.button("Skip Letter (N)", use_container_width=True):
            if ctx.video_processor:
                ctx.video_processor.skip_letter_action()
    with c2:
        if st.button("Skip Word (S)", use_container_width=True):
            if ctx.video_processor:
                ctx.video_processor.skip_word_action()
    
    st.markdown("---")
    if st.button("Reset Progress", type="primary"):
        if ctx.video_processor:
            ctx.video_processor.reset_progress()
    
    st.info("""
    **Instructions:**
    1.  Allow camera access.
    2.  Spell the word shown in yellow.
    3.  Hold the sign until the green bar fills up.
    4.  Use buttons above to skip actions.
    """)
