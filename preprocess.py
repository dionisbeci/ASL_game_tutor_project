import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Data Path Configuration
# The user specified paths starting with '../data', but based on the project structure, 
# 'data' is a subdirectory of the project root where this script resides.
# We will check both locations to ensure robustness.

def get_data_paths():
    """Determine the correct paths for raw and processed data."""
    # Try the user's specified path first
    user_raw_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "asl_alphabet_train"))
    user_processed_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))
    
    # Fallback to internal project path if user's path doesn't exist
    internal_raw_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "raw", "asl_alphabet_train"))
    internal_processed_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "processed"))
    
    # We also noticed a nested folder in the raw directory
    nested_raw_path_user = os.path.join(user_raw_path, "asl_alphabet_train")
    nested_raw_path_internal = os.path.join(internal_raw_path, "asl_alphabet_train")

    if os.path.exists(nested_raw_path_user):
        return nested_raw_path_user, user_processed_path
    if os.path.exists(user_raw_path):
        return user_raw_path, user_processed_path
    if os.path.exists(nested_raw_path_internal):
        return nested_raw_path_internal, internal_processed_path
    return internal_raw_path, internal_processed_path

def preprocess_data():
    raw_path, processed_path = get_data_paths()
    
    if not os.path.exists(raw_path):
        print(f"Error: Raw data directory not found at {raw_path}")
        return

    # Ensure processed directory exists
    os.makedirs(processed_path, exist_ok=True)

    X = []
    y = []
    
    # Loop through every class folder (A-Z, space, del, nothing)
    classes = sorted([d for d in os.listdir(raw_path) if os.path.isdir(os.path.join(raw_path, d))])
    
    print(f"Loading images from: {raw_path}")
    print(f"Found {len(classes)} classes: {', '.join(classes)}")
    
    for label in classes:
        label_path = os.path.join(raw_path, label)
        print(f"Processing class: {label}...", end="\r")
        
        # Load every image using OpenCV
        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Resize image to 64x64 pixels
            img = cv2.resize(img, (64, 64))
            
            # Convert image from BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Normalize pixel values by dividing by 255.0
            img = img.astype('float32') / 255.0
            
            X.append(img)
            y.append(label)
    
    print("\nData loading complete. Converting to NumPy arrays...")
    X = np.array(X)
    y = np.array(y)
    
    # Use LabelEncoder to turn letters into numbers (A->0, B->1)
    print("Encoding labels...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Store the classes list so we know which number maps to which letter
    label_map = le.classes_
    
    # to_categorical for one-hot encoding using numpy
    print("Encoding labels...")
    num_classes = len(label_map)
    y_categorical = np.eye(num_classes)[y_encoded]
    
    # Split the data: 80% for Training, 20% for Testing
    print("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Output: Save 4 files to the processed folder
    print(f"Saving processed data to: {processed_path}")
    np.save(os.path.join(processed_path, "X_train.npy"), X_train)
    np.save(os.path.join(processed_path, "X_test.npy"), X_test)
    np.save(os.path.join(processed_path, "y_train.npy"), y_train)
    np.save(os.path.join(processed_path, "y_test.npy"), y_test)
    np.save(os.path.join(processed_path, "label_map.npy"), label_map)
    
    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    preprocess_data()
