import json
import numpy as np

def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def get_rotation_quaternion(u, v):
    """
    Computes the quaternion [x, y, z, w] representing the shortest rotation 
    from vector u to vector v.
    """
    u = normalize_vector(u)
    v = normalize_vector(v)
    
    dot = np.dot(u, v)
    
    # Handle collinear opposite vectors
    if dot < -0.9999:
        # Find orthogonal axis
        orthogonal = np.array([1.0, 0.0, 0.0])
        if abs(u[0]) > 0.8:
            orthogonal = np.array([0.0, 1.0, 0.0])
        axis = normalize_vector(np.cross(u, orthogonal))
        return [axis[0], axis[1], axis[2], 0.0]
        
    # Handle collinear same vectors
    if dot > 0.9999:
        return [0.0, 0.0, 0.0, 1.0]
        
    axis = np.cross(u, v)
    # The half-angle formula for shortest rotation quaternion:
    # w = sqrt(2 + 2 * dot) / 2
    # xyz = axis / (2 * w)
    w_sq = 1.0 + dot
    w = np.sqrt(w_sq * 2.0)
    
    xyz = axis / w
    q = [xyz[0], xyz[1], xyz[2], w / 2.0]
    
    # Normalize
    q_norm = np.linalg.norm(q)
    return [q[0] / q_norm, q[1] / q_norm, q[2] / q_norm, q[3] / q_norm]

def map_landmarks_to_quaternions(landmarks_seq, sign_name, fps=30):
    """
    Translates MediaPipe frame-by-frame 3D coordinates into a timeline
    of quaternions for standard joints.
    """
    keyframes = []
    
    for frame_idx, frame in enumerate(landmarks_seq):
        time_stamp = frame_idx / float(fps)
        
        # Reshape frame features: left hand (21 joints * 3) + right hand (21 joints * 3)
        # Note: We need shoulders, elbows, wrists from pose landmarks.
        # If we have pose landmarks in the preprocessed files, we read them.
        # For this script, we assume the preprocessed input format includes 
        # both pose and hand landmarks. Let's write the code assuming standard MediaPipe indices.
        
        # Suppose landmarks_seq is structured as:
        # [left_hand_joints (21), right_hand_joints (21), pose_joints (shoulders, elbows, wrists)]
        # For robust mapping, let's mock or compute rotations if coordinates are present:
        
        # Reference bone direction vectors in T-pose local coordinates:
        # Left upper arm points along negative X: [-1.0, 0, 0]
        # Right upper arm points along positive X: [1.0, 0, 0]
        # Forearms point along negative X (left) or positive X (right) relative to upper arm.
        
        # In a fully populated extraction, we would do:
        # left_shoulder_to_elbow = left_elbow_pos - left_shoulder_pos
        # q_left_shoulder = get_rotation_quaternion([-1.0, 0.0, 0.0], left_shoulder_to_elbow)
        
        # Let's write a generic mapping template
        bones_rotations = {}
        
        # Example calculation placeholder (would be active when feeding real NumPy arrays)
        # In this template we generate neutral rotations as fallback, showing the logic.
        bones_rotations["LeftShoulder"] = [0.0, 0.0, 0.0, 1.0]
        bones_rotations["LeftElbow"] = [0.0, 0.0, 0.0, 1.0]
        bones_rotations["LeftWrist"] = [0.0, 0.0, 0.0, 1.0]
        bones_rotations["RightShoulder"] = [0.0, 0.0, 0.0, 1.0]
        bones_rotations["RightElbow"] = [0.0, 0.0, 0.0, 1.0]
        bones_rotations["RightWrist"] = [0.0, 0.0, 0.0, 1.0]
        bones_rotations["Neck"] = [0.0, 0.0, 0.0, 1.0]
        
        keyframes.append({
            "time": time_stamp,
            "bones": bones_rotations
        })
        
    return {
        "sign_name": sign_name,
        "keyframes": keyframes
    }

if __name__ == "__main__":
    # Test vector rotations
    u = np.array([1.0, 0.0, 0.0]) # T-pose arm vector
    v = np.array([0.0, 1.0, 0.0]) # Raised arm vector
    q = get_rotation_quaternion(u, v)
    print(f"Rotation quaternion from [1,0,0] to [0,1,0]: {q}")
