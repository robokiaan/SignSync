import os
import cv2
import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic

def normalize_hand_landmarks(hand_landmarks):
    """
    Translates hand coordinates relative to the wrist (making it 0,0,0)
    and scales them by the distance between the wrist (index 0) and
    the middle finger knuckle (index 9).
    """
    if hand_landmarks is None:
        return np.zeros((21, 3))
        
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    
    # Translate relative to wrist (index 0)
    wrist = landmarks[0]
    translated = landmarks - wrist
    
    # Calculate scale factor: distance between wrist (0) and middle knuckle (9)
    middle_knuckle = translated[9]
    scale_factor = np.linalg.norm(middle_knuckle)
    
    if scale_factor > 0:
        normalized = translated / scale_factor
    else:
        normalized = translated
        
    return normalized

def extract_landmarks_from_video(video_path):
    """
    Runs MediaPipe Holistic on a single video file and returns a sequence
    of normalized landmarks for each frame.
    """
    cap = cv2.VideoCapture(video_path)
    sequence_data = []
    
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert color space for MediaPipe (BGR -> RGB)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = holistic.process(image_rgb)
            
            # Extract Left Hand (21 joints * 3 = 63 values)
            left_hand_norm = normalize_hand_landmarks(results.left_hand_landmarks)
            
            # Extract Right Hand (21 joints * 3 = 63 values)
            right_hand_norm = normalize_hand_landmarks(results.right_hand_landmarks)
            
            # Combined features for both hands (2 * 21 * 3 = 126 coordinates)
            frame_features = np.concatenate([left_hand_norm.flatten(), right_hand_norm.flatten()])
            sequence_data.append(frame_features)
            
    cap.release()
    return np.array(sequence_data)

def preprocess_dataset(dataset_dir, output_dir):
    """
    Iterates through dataset folders representing classes, runs MediaPipe,
    and saves .npy files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Each subdirectory inside dataset_dir is a sign class
    classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    print(f"Discovered {len(classes)} classes in dataset directory: {classes}")
    
    for cls in classes:
        cls_dir = os.path.join(dataset_dir, cls)
        cls_out_dir = os.path.join(output_dir, cls)
        os.makedirs(cls_out_dir, exist_ok=True)
        
        videos = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.mp4', '.avi', '.mkv'))]
        print(f"Processing class '{cls}': found {len(videos)} videos.")
        
        for v in videos:
            video_path = os.path.join(cls_dir, v)
            output_filename = os.path.splitext(v)[0] + "_landmarks.npy"
            output_path = os.path.join(cls_out_dir, output_filename)
            
            # Skip if already preprocessed
            if os.path.exists(output_path):
                continue
                
            print(f"  -> Preprocessing {v}...")
            try:
                landmarks = extract_landmarks_from_video(video_path)
                if len(landmarks) > 0:
                    np.save(output_path, landmarks)
                    print(f"     Saved shape: {landmarks.shape}")
                else:
                    print(f"     [Warning] No landmarks detected in {v}.")
            except Exception as e:
                print(f"     [Error] Failed to process {v}: {e}")

if __name__ == "__main__":
    # Example usage:
    # Change these paths to point to your INCLUDE videos dataset location
    DATASET_PATH = "./dataset"
    OUTPUT_PATH = "./preprocessed_landmarks"
    
    if os.path.exists(DATASET_PATH):
        preprocess_dataset(DATASET_PATH, OUTPUT_PATH)
    else:
        print(f"Please create a '{DATASET_PATH}' directory containing folders of sign videos to run preprocessing.")
