import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import os
import sys

# Add the current directory to sys.path to allow importing tutor_logic
sys.path.append(os.path.dirname(__file__))
from tutor_logic import TutorAgent

# --- Configuration ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "mlp_model.h5")
LABEL_MAP_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "label_map_mlp.npy")

# --- Resource Loading ---
@st.cache_resource
def load_model_and_labels():
    """Loads the Keras model and label map (cached)."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at {MODEL_PATH}")
        return None, None
    
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        # Load label map and ensure it allows pickle
        label_map = np.load(LABEL_MAP_PATH, allow_pickle=True)
        # Invert if necessary (index -> string)
        # Assuming label_map is just an array of strings where index matches class_id
        return model, label_map
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return None, None

model, class_labels = load_model_and_labels()

# --- Initializes MediaPipe ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
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

# --- Video Grabbing & Processing ---
class ASLProcessor(VideoProcessorBase):
    def __init__(self):
        self.debounce_counter = 0
        self.min_threshold_frames = 10
        self.last_prediction = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # 1. Mirror Flip
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        
        # 2. Hand Tracking
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_img)
        
        predicted_letter = "?"
        confidence = 0.0
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # 3. Feature Extraction
                landmarks = []
                wrist_x = hand_landmarks.landmark[0].x
                wrist_y = hand_landmarks.landmark[0].y
                
                for lm in hand_landmarks.landmark:
                    # Normalize relative to wrist
                    landmarks.append(lm.x - wrist_x)
                    landmarks.append(lm.y - wrist_y)
                
                # 4. Prediction
                if model is not None:
                    input_data = np.array([landmarks], dtype=np.float32)
                    preds = model.predict(input_data, verbose=0)
                    class_idx = np.argmax(preds)
                    confidence = preds[0][class_idx]
                    
                    if class_labels is not None and class_idx < len(class_labels):
                        predicted_letter = class_labels[class_idx]
        
        # 5. Game Logic
        target_word = st.session_state['current_word']
        target_letter = target_word[st.session_state['word_idx']].upper() if target_word else ""
        
        # Check prediction
        if predicted_letter == target_letter and confidence > 0.8:
            self.debounce_counter += 1
        else:
            self.debounce_counter = 0
            
        # Logic update (must be careful editing session state from thread, usually safe for reads/basic writes)
        # Note: streamlit-webrtc runs in a separate thread. Direct session_state modification can be tricky.
        # But for simple counters/flags it often works or use a queue. 
        # For this requirement, we'll try direct access as it's the simplest approach requested.
        
        if self.debounce_counter >= self.min_threshold_frames:
            self.debounce_counter = 0
            # Success!
            # We can't easily call agent methods that change internal state safely if they aren't thread safe, 
            # but here we update session state proxy variables ideally.
            # However, user asked to "Call agent.update_performance()"
            st.session_state['agent'].update_performance(target_letter, True)
            
            # Advance letter
            st.session_state['word_idx'] += 1
            if st.session_state['word_idx'] >= len(target_word):
                st.session_state['score'] += 1
                st.session_state['current_word'] = st.session_state['agent'].get_next_word()
                st.session_state['word_idx'] = 0
        
        # 6. UI Overlay
        # Top Bar
        cv2.rectangle(img, (0, 0), (w, 80), (0, 0, 0), -1)
        
        # Target Word
        word_display = ""
        for i, char in enumerate(target_word):
            if i == st.session_state['word_idx']:
                word_display += f"[{char}] "
            else:
                word_display += f"{char} "
                
        cv2.putText(img, f"TARGET: {word_display}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        # Score & Prediction
        cv2.putText(img, f"Score: {st.session_state['score']}", (w - 200, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
        # Debug Prediction
        if predicted_letter != "?":
            color = (0, 255, 0) if predicted_letter == target_letter else (0, 0, 255)
            cv2.putText(img, f"Pred: {predicted_letter} ({confidence:.2f})", (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
        # Progress Bar for Debounce
        if self.debounce_counter > 0:
            bar_width = int((self.debounce_counter / self.min_threshold_frames) * 100)
            cv2.rectangle(img, (20, 90), (20 + bar_width, 100), (0, 255, 255), -1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Layout ---
st.set_page_config(layout="wide", page_title="ASL Game Tutor")

st.title("ASL Game Tutor - Cloud Edition")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Webcam Feed")
    ctx = webrtc_streamer(
        key="asl-tutor",
        video_processor_factory=ASLProcessor,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

with col2:
    st.markdown("### Instructions")
    st.info("1. Allow webcam access.\n2. Spell the target word letter by letter.\n3. Hold the sign for a moment to confirm.")
    
    st.markdown("### Performance")
    mistakes = st.session_state['agent'].mistake_count
    st.bar_chart(mistakes)
    
    if st.button("Reset Progress"):
        st.session_state['score'] = 0
        st.session_state['agent'] = TutorAgent(words_file_path=os.path.join(os.path.dirname(__file__), "words.txt"))
        st.session_state['current_word'] = st.session_state['agent'].get_next_word()
        st.session_state['word_idx'] = 0
        st.rerun()
