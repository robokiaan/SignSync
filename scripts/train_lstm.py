import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, Input
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def load_preprocessed_dataset(landmarks_dir, sequence_len=30, num_features=126):
    """
    Loads all .npy files from class folders, pads/truncates sequences
    to sequence_len, and returns dataset arrays X and y.
    """
    X = []
    y = []
    
    classes = [d for d in os.listdir(landmarks_dir) if os.path.isdir(os.path.join(landmarks_dir, d))]
    class_to_id = {cls: idx for idx, cls in enumerate(classes)}
    
    print(f"Loading dataset. Total classes: {len(classes)}")
    
    for cls in classes:
        cls_dir = os.path.join(landmarks_dir, cls)
        files = [f for f in os.listdir(cls_dir) if f.endswith(".npy")]
        
        for f in files:
            file_path = os.path.join(cls_dir, f)
            # Load frame-by-frame landmarks (shape: [num_frames, 126])
            seq = np.load(file_path)
            
            # Pad or truncate to fixed sequence length (e.g. 30 frames)
            if len(seq) < sequence_len:
                # Pad with zeros at the end
                padding = np.zeros((sequence_len - len(seq), num_features))
                seq_padded = np.concatenate([seq, padding], axis=0)
            else:
                # Truncate
                seq_padded = seq[:sequence_len]
                
            X.append(seq_padded)
            y.append(class_to_id[cls])
            
    return np.array(X), np.array(y), classes

def build_lstm_model(num_classes, sequence_len=30, num_features=126):
    """
    Constructs the Keras sequence classification model.
    """
    model = Sequential([
        Input(shape=(sequence_len, num_features)),
        Masking(mask_value=0.0), # Ignore padding frames
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    LANDMARKS_DIR = "./preprocessed_landmarks"
    MODEL_OUT_H5 = "./lstm_model.h5"
    TFJS_OUT_DIR = "../app/static/model"
    
    if not os.path.exists(LANDMARKS_DIR):
        print(f"Error: Landmarked directory '{LANDMARKS_DIR}' not found. Run preprocess.py first.")
        return
        
    X, y, classes = load_preprocessed_dataset(LANDMARKS_DIR)
    if len(X) == 0:
        print("No samples found. Preprocessing might have failed or skipped.")
        return
        
    num_classes = len(classes)
    y_one_hot = to_categorical(y, num_classes=num_classes)
    
    # Split into train & test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_one_hot, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train samples: {X_train.shape}, Test samples: {X_test.shape}")
    
    # Build and Train Model
    model = build_lstm_model(num_classes)
    model.summary()
    
    print("Starting LSTM training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=30,
        batch_size=8
    )
    
    # Save standard Keras model
    model.save(MODEL_OUT_H5)
    print(f"Keras model saved to {MODEL_OUT_H5}")
    
    # Compilation code for TensorFlow.js converter
    print("\n--- To convert this model to TensorFlow.js format, run: ---")
    print(f"tensorflowjs_converter --input_format=keras {MODEL_OUT_H5} {TFJS_OUT_DIR}")
    
    # Try to convert programmatically if tensorflowjs package is installed
    try:
        import tensorflowjs as tfjs
        tfjs.converters.save_keras_model(model, TFJS_OUT_DIR)
        print(f"Successfully converted and saved TF.js model files to {TFJS_OUT_DIR}")
    except ImportError:
        print("tensorflowjs module not installed. Run 'pip install tensorflowjs' to enable automatic TF.js conversion.")

if __name__ == "__main__":
    main()
