import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def train_model():
    # 1. Configuration
    data_path = os.path.join("data", "processed")
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    # 2. Load the processed data
    print("Loading preprocessed data from:", data_path)
    try:
        X_train = np.load(os.path.join(data_path, "X_train.npy"))
        X_test = np.load(os.path.join(data_path, "X_test.npy"))
        y_train = np.load(os.path.join(data_path, "y_train.npy"))
        y_test = np.load(os.path.join(data_path, "y_test.npy"))
    except FileNotFoundError as e:
        print(f"Error: Could not find processed data. Please run preprocess.py first. Details: {e}")
        return

    print(f"Dataset Loaded:")
    print(f" - Train images: {X_train.shape}")
    print(f" - Test images: {X_test.shape}")
    print(f" - Classes: {y_train.shape[1]}")

    # 3. Build the CNN Model (Based on Project Proposal)
    # Architecture: Conv2D -> MaxPooling -> Dropout -> Dense
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        layers.Dropout(0.25),

        # Flatten and Fully Connected
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(y_train.shape[1], activation='softmax')
    ])

    # 4. Compile the Model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # 5. Training
    print("\nStarting model training...")
    EPOCHS = 10
    BATCH_SIZE = 64

    # Early stopping to prevent overfitting if accuracy stops improving
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=[early_stop]
    )

    # 6. Save Model and Weights
    model_save_path = os.path.join(model_dir, "asl_model.h5")
    model.save(model_save_path)
    print(f"\nModel training complete! Saved to: {model_save_path}")

    # 7. Visualize Training Results
    plt.figure(figsize=(12, 5))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'training_history.png'))
    print(f"Training history plot saved to: {os.path.join(model_dir, 'training_history.png')}")
    plt.show()

if __name__ == "__main__":
    train_model()
