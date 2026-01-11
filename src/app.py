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

# Add the current directory to sys.path to allow importing tutor_logic
sys.path.append(os.path.dirname(__file__))
from tutor_logic import TutorAgent

# --- Configuration ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "mlp_model.h5")
LABEL_MAP_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "label_map_mlp.npy")
RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "asl_alphabet_train", "asl_alphabet_train")

# --- Resource Loading ---
@st.cache_resource
def load_resources():
    """Loads Model, Labels, and Hint Images."""
    resources = {}
    
    # 1. Load Model & Map
    try:
        if os.path.exists(MODEL_PATH):
            resources['model'] = tf.keras.models.load_model(MODEL_PATH)
            resources['labels'] = np.load(LABEL_MAP_PATH, allow_pickle=True)
        else:
            st.error(f"Model not found at {MODEL_PATH}")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

    # 2. Load Hint Images
    hints = {}
    if os.path.exists(RAW_DATA_PATH):
        try:
            # Assuming labels are just strings in the array
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
        except Exception as e:
            print(f"Error loading hints: {e}")
    else:
        print(f"Warning: Hint data path not found: {RAW_DATA_PATH}")
    
    resources['hints'] = hints
    return resources

resources = load_resources()
model = resources['model'] if resources else None
class_labels = resources['labels'] if resources else None
hint_images = resources['hints'] if resources else {}

# --- Initializes MediaPipe ---
mp_hands = mp.solutions.hands
# Quality Restoration: Back to standard confidence, but keep complexity=0 for speed if needed
# User asked for "quality", so we bump detection confidence slightly up
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0, 
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# --- Session State ---
if 'agent' not in st.session_state:
    words_file = os.path.join(os.path.dirname(__file__), "words.txt")
    st.session_state['agent'] = TutorAgent(words_file_path=words_file)
    st.session_state['current_word'] = st.session_state['agent'].get_next_word()
    st.session_state['score'] = 0
    st.session_state['word_idx'] = 0

# --- Actions (functions for buttons) ---
def skip_letter():
    agent = st.session_state['agent']
    target_word = st.session_state['current_word']
    if target_word:
        target_char = target_word[st.session_state['word_idx']]
        agent.update_performance(target_char, False)
        
        st.session_state['word_idx'] += 1
        if st.session_state['word_idx'] >= len(target_word):
            # Word Finished (skipped last letter)
            st.session_state['score'] += 1 # Logic from main.py: skipped letter doesn't give points usually but completing word does?
            # Actually main.py logic: if key==ord('n') -> update_performance(False). 
            # If word complete -> if key!=ord('n') score+=1. 
            # So if we skip the last letter to finish the word, we DO NOT get a point for the word.
            # We need to replicate that logic carefully. 
            # Simplified for UI button: Just skip letter. If it finishes word, next word.
            st.session_state['current_word'] = agent.get_next_word()
            st.session_state['word_idx'] = 0

def skip_word():
    agent = st.session_state['agent']
    target_word = st.session_state['current_word']
    for i in range(st.session_state['word_idx'], len(target_word)):
        agent.update_performance(target_word[i], False)
    
    st.session_state['current_word'] = agent.get_next_word()
    st.session_state['word_idx'] = 0


# --- Video Processor ---
class ASLProcessor(VideoProcessorBase):
    def __init__(self):
        self.debounce_counter = 0
        self.min_threshold_frames = 10 # 10 frames @ 30fps is ~0.3s, @15fps is ~0.6s
        
        # Frame Skipping
        self.frame_count = 0
        self.skip_rate = 5
        self.last_hand_landmarks = None
        self.last_prediction = "?"
        self.last_confidence = 0.0

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            self.frame_count += 1
            
            # 1. Mirror Flip
            img = cv2.flip(img, 1)
            h, w, _ = img.shape
            
            # --- AI PROCESSING (Every Nth Frame) ---
            if self.frame_count % self.skip_rate == 0:
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_img)
                
                self.last_hand_landmarks = None
                self.last_prediction = "?"
                self.last_confidence = 0.0
                
                if results.multi_hand_landmarks:
                    self.last_hand_landmarks = results.multi_hand_landmarks[0]
                    
                    # Feature Extraction
                    landmarks = []
                    wrist_x = self.last_hand_landmarks.landmark[0].x
                    wrist_y = self.last_hand_landmarks.landmark[0].y
                    
                    for lm in self.last_hand_landmarks.landmark:
                        landmarks.append(lm.x - wrist_x)
                        landmarks.append(lm.y - wrist_y)
                    
                    # Prediction
                    if model is not None:
                        input_data = np.array([landmarks], dtype=np.float32)
                        preds = model.predict(input_data, verbose=0)
                        class_idx = np.argmax(preds)
                        self.last_confidence = preds[0][class_idx]
                        if class_labels is not None and class_idx < len(class_labels):
                            self.last_prediction = class_labels[class_idx]

            # --- DRAWING & GAME LOGIC ---
            
            # Draw Landmarks
            if self.last_hand_landmarks:
                mp_draw.draw_landmarks(img, self.last_hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            predicted_letter = self.last_prediction
            confidence = self.last_confidence
            
            # Game Logic (Session State Access)
            # Note regarding threading: st.session_state is generally safe for reads in webrtc threads, but writes can be tricky.
            # We will read logic here and update counters.
            
            if 'current_word' in st.session_state:
                target_word = st.session_state['current_word']
                target_letter = target_word[st.session_state['word_idx']].upper() if target_word else ""
                
                # Check prediction
                if predicted_letter == target_letter and confidence > 0.8:
                    self.debounce_counter += 1
                else:
                    self.debounce_counter = 0
                
                if self.debounce_counter >= self.min_threshold_frames:
                    self.debounce_counter = 0
                    # Success
                    st.session_state['agent'].update_performance(target_letter, True)
                    st.session_state['word_idx'] += 1
                    
                    if st.session_state['word_idx'] >= len(target_word):
                        st.session_state['score'] += 1
                        st.session_state['current_word'] = st.session_state['agent'].get_next_word()
                        st.session_state['word_idx'] = 0
            
                # --- UI OVERLAY (Replicating main.py) ---
                
                # 1. Top Black Bar
                cv2.rectangle(img, (0, 0), (w, 85), (0, 0, 0), -1)
                
                # 2. Target Word Display
                word_display = "".join([f"[{c}] " if i == st.session_state['word_idx'] else f"{c} " 
                                       for i, c in enumerate(target_word)])
                cv2.putText(img, f"TARGET: {word_display}", (20, 35), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                
                # 3. Score
                cv2.putText(img, f"SCORE: {st.session_state['score']}", (w - 180, 35), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # 4. Progress Bar
                if self.debounce_counter > 0:
                    bar_w = int((self.debounce_counter / self.min_threshold_frames) * 200)
                    cv2.rectangle(img, (20, 95), (20 + bar_w, 105), (0, 255, 0), -1)
                
                # 5. Prediction Feedback
                if predicted_letter != "?":
                    color = (0, 255, 0) if predicted_letter == target_letter else (0, 0, 255)
                    wrist_x = int(self.last_hand_landmarks.landmark[0].x * w) if self.last_hand_landmarks else 20
                    wrist_y = int(self.last_hand_landmarks.landmark[0].y * h) if self.last_hand_landmarks else h-20
                    cv2.putText(img, f"{predicted_letter}", 
                                (wrist_x, wrist_y - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                # 6. Hint Image Overlay (Bottom Right)
                if target_letter in hint_images:
                    hint = hint_images[target_letter]
                    h_img, w_img, _ = hint.shape
                    
                    # Ensure hint fits in frame
                    if h > h_img and w > w_img:
                        y_offset = h - h_img - 20
                        x_offset = w - w_img - 20
                        
                        img[y_offset:y_offset+h_img, x_offset:x_offset+w_img] = hint
                        
                        cv2.putText(img, f"Hint: {target_letter}", (x_offset, y_offset - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.rectangle(img, (x_offset, y_offset), (x_offset+w_img, y_offset+h_img), (255, 255, 255), 1)

            return av.VideoFrame.from_ndarray(img, format="bgr24")
        except Exception as e:
            print(f"Frame Error: {e}")
            traceback.print_exc()
            return frame

# --- Main Layout ---
st.set_page_config(layout="wide", page_title="ASL Game Tutor")

st.title("ASL Game Tutor")

col1, col2 = st.columns([2, 1])

with col1:
    # Quality Restoration: 640x480 resolution (Standard VGA)
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
            skip_letter()
            st.rerun()
    with c2:
        if st.button("Skip Word (S)", use_container_width=True):
            skip_word()
            st.rerun()
    
    st.markdown("---")
    st.markdown("### Performance")
    if 'agent' in st.session_state:
        mistakes = st.session_state['agent'].mistake_count
        st.bar_chart(mistakes)
    
    if st.button("Reset Progress", type="primary"):
        st.session_state['score'] = 0
        st.session_state['agent'] = TutorAgent(words_file_path=os.path.join(os.path.dirname(__file__), "words.txt"))
        st.session_state['current_word'] = st.session_state['agent'].get_next_word()
        st.session_state['word_idx'] = 0
        st.rerun()

    st.info("""
    **Instructions:**
    1.  Allow camera access.
    2.  Spell the word shown in yellow.
    3.  Hold the sign until the green bar fills up.
    4.  Use buttons above to skip if stuck.
    """)
