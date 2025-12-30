import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os

# --- Configuration ---
DATA_PATH = os.path.join("..", "data", "processed", "data.csv")
MODEL_SAVE_PATH = "../models/mlp_model.h5"
LABEL_MAP_PATH = "../models/label_map_mlp.npy"

def train_mlp():
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found. Please run create_dataset.py first.")
        return

    # 1. Load Data
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    # 2. Separate Features and Labels
    X = df.drop('label', axis=1).values
    y_raw = df['label'].values

    # 3. Encode Labels
    print("Encoding labels...")
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    # Save label mapping for inference later
    np.save(LABEL_MAP_PATH, le.classes_)
    print(f"Label mapping saved to {LABEL_MAP_PATH}")

    # 4. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Build MLP Model (Week 8 Syllabus style)
    print("Building MLP Model...")
    model = models.Sequential([
        layers.Input(shape=(42,)), # 21 landmarks * 2 (x,y)
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(29, activation='softmax') # 29 ASL classes
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 6. Train the Model
    print("Starting training...")
    history = model.fit(X_train, y_train, 
                        epochs=20, 
                        batch_size=32, 
                        validation_data=(X_test, y_test))

    # 7. Evaluate and Save
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"\nFinal Test Accuracy: {test_acc*100:.2f}%")

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    # 8. Plot Accuracy vs Epochs
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('MLP Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('mlp_training_history.png')
    print("Training plot saved as mlp_training_history.png")
    plt.show()

if __name__ == "__main__":
    train_mlp()
