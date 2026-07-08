import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import engine, Base, SessionLocal
from app.models import SignDictionary, Lesson, LessonItem
import math

def get_quaternion_from_euler(x_deg, y_deg, z_deg):
    """
    Computes a quaternion [x, y, z, w] from Euler angles in degrees (roll, pitch, yaw).
    Uses standard Three.js XYZ Euler order.
    """
    rx = math.radians(x_deg)
    ry = math.radians(y_deg)
    rz = math.radians(z_deg)
    
    cx = math.cos(rx / 2.0)
    sx = math.sin(rx / 2.0)
    cy = math.cos(ry / 2.0)
    sy = math.sin(ry / 2.0)
    cz = math.cos(rz / 2.0)
    sz = math.sin(rz / 2.0)
    
    # Three.js XYZ order formula:
    x = sx * cy * cz + cx * sy * sz
    y = cx * sy * cz - sx * cy * sz
    z = cx * cy * sz + sx * sy * cz
    w = cx * cy * cz - sx * sy * sz
    
    return [x, y, z, w]

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import engine, Base, SessionLocal
from app.models import SignDictionary, Lesson, LessonItem
import math
import numpy as np

def get_quaternion_from_euler(x_deg, y_deg, z_deg):
    """
    Computes a quaternion [x, y, z, w] from Euler angles in degrees (roll, pitch, yaw).
    Uses standard Three.js XYZ Euler order.
    """
    rx = math.radians(x_deg)
    ry = math.radians(y_deg)
    rz = math.radians(z_deg)
    
    cx = math.cos(rx / 2.0)
    sx = math.sin(rx / 2.0)
    cy = math.cos(ry / 2.0)
    sy = math.sin(ry / 2.0)
    cz = math.cos(rz / 2.0)
    sz = math.sin(rz / 2.0)
    
    # Three.js XYZ order formula:
    x = sx * cy * cz + cx * sy * sz
    y = cx * sy * cz - sx * cy * sz
    z = cx * cy * sz + sx * sy * cz
    w = cx * cy * cz - sx * sy * sz
    
    return [x, y, z, w]

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def get_rotation_quaternion(u, v):
    u = normalize(u)
    v = normalize(v)
    dot = np.dot(u, v)
    if dot < -0.9999:
        orthogonal = np.array([1.0, 0.0, 0.0])
        if abs(u[0]) > 0.8:
            orthogonal = np.array([0.0, 1.0, 0.0])
        axis = normalize(np.cross(u, orthogonal))
        return [float(axis[0]), float(axis[1]), float(axis[2]), 0.0]
    if dot > 0.9999:
        return [0.0, 0.0, 0.0, 1.0]
    axis = np.cross(u, v)
    w = np.sqrt((1.0 + dot) * 2.0)
    xyz = axis / w
    q = [float(xyz[0]), float(xyz[1]), float(xyz[2]), float(w / 2.0)]
    q_norm = np.linalg.norm(q)
    return [float(q[0]/q_norm), float(q[1]/q_norm), float(q[2]/q_norm), float(q[3]/q_norm)]

def q_conjugate(q):
    return [-q[0], -q[1], -q[2], q[3]]

def q_mult(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return [x, y, z, w]

def q_rot(q, v):
    v_q = [v[0], v[1], v[2], 0.0]
    res = q_mult(q_mult(q, v_q), q_conjugate(q))
    return np.array(res[:3])

def solve_arm_ik(shoulder_pos, wrist_target, elbow_hint, is_left=True, hand_dir=[0.0, 1.0, 0.0], palm_dir=[1.0, 0.0, 0.0]):
    L1 = 0.45
    L2 = 0.45
    
    S = np.array(shoulder_pos)
    W = np.array(wrist_target)
    E_hint = np.array(elbow_hint)
    
    SW = W - S
    d = np.linalg.norm(SW)
    if d > (L1 + L2) * 0.99:
        d = (L1 + L2) * 0.99
        SW = normalize(SW) * d
        W = S + SW
        
    cos_alpha = (L1**2 + d**2 - L2**2) / (2 * L1 * d)
    cos_alpha = max(-1.0, min(1.0, cos_alpha))
    alpha = np.arccos(cos_alpha)
    
    sw_dir = normalize(SW)
    proj = S + sw_dir * (L1 * cos_alpha)
    eh_proj = E_hint - proj
    perp_dir = normalize(eh_proj - np.dot(eh_proj, sw_dir) * sw_dir)
    
    height = L1 * np.sin(alpha)
    E = proj + perp_dir * height
    
    # 1. Shoulder rotation
    ref_upper = np.array([-1.0, 0.0, 0.0]) if is_left else np.array([1.0, 0.0, 0.0])
    v_upper = E - S
    q_shoulder = get_rotation_quaternion(ref_upper, v_upper)
    
    # 2. Elbow rotation
    v_fore = W - E
    q_sh_inv = q_conjugate(q_shoulder)
    v_fore_local = q_rot(q_sh_inv, v_fore)
    ref_fore = np.array([-1.0, 0.0, 0.0]) if is_left else np.array([1.0, 0.0, 0.0])
    q_elbow = get_rotation_quaternion(ref_fore, v_fore_local)
    
    # 3. Wrist rotation
    q_se = q_mult(q_shoulder, q_elbow)
    q_se_inv = q_conjugate(q_se)
    
    target_hand_dir_global = np.array(hand_dir)
    target_palm_normal_global = np.array(palm_dir)
    
    target_hand_dir_local = q_rot(q_se_inv, target_hand_dir_global)
    target_palm_normal_local = q_rot(q_se_inv, target_palm_normal_global)
    
    q_w_dir = get_rotation_quaternion(ref_fore, target_hand_dir_local)
    
    ref_palm = np.array([0.0, -1.0, 0.0])
    palm_local_temp = q_rot(q_w_dir, ref_palm)
    q_w_twist = get_rotation_quaternion(palm_local_temp, target_palm_normal_local)
    
    q_wrist = q_mult(q_w_twist, q_w_dir)
    
    return [float(x) for x in q_shoulder], [float(x) for x in q_elbow], [float(x) for x in q_wrist]

def generate_namaste_keyframes():
    # Namaste: Both hands swing forward/inward, elbows bend,
    # and palms meet flat at the chest pointing upward.
    neutral = [0.0, 0.0, 0.0, 1.0]
    l_sh_neutral = [0.0, 0.0, 0.7071, 0.7071]
    r_sh_neutral = [0.0, 0.0, -0.7071, 0.7071]
    
    # Left Arm solved IK values
    l_sh, l_el, l_wr = solve_arm_ik(
        shoulder_pos=[-0.4, 0.95, 0.0],
        wrist_target=[-0.015, 0.60, 0.35],
        elbow_hint=[-0.5, 0.52, 0.15],
        is_left=True,
        hand_dir=[0.0, 1.0, 0.0],
        palm_dir=[1.0, 0.0, 0.0]
    )
    
    # Right Arm solved IK values
    r_sh, r_el, r_wr = solve_arm_ik(
        shoulder_pos=[0.4, 0.95, 0.0],
        wrist_target=[0.015, 0.60, 0.35],
        elbow_hint=[0.5, 0.52, 0.15],
        is_left=False,
        hand_dir=[0.0, 1.0, 0.0],
        palm_dir=[-1.0, 0.0, 0.0]
    )
    
    # Head bow
    neck_bow = get_quaternion_from_euler(15, 0, 0)
    
    return [
        {
            "time": 0.0,
            "bones": {
                "LeftShoulder": l_sh_neutral, "LeftElbow": neutral, "LeftWrist": neutral,
                "RightShoulder": r_sh_neutral, "RightElbow": neutral, "RightWrist": neutral,
                "Neck": neutral
            }
        },
        {
            "time": 0.5,
            # Intermediate pose moving towards chest
            "bones": {
                "LeftShoulder": [0.0, 0.136, 0.315, 0.939],
                "LeftElbow": [0.0, 0.264, 0.243, 0.924],
                "LeftWrist": [0.260, 0.300, -0.132, 0.886],
                "RightShoulder": [0.0, -0.136, -0.315, 0.939],
                "RightElbow": [0.0, -0.264, -0.243, 0.924],
                "RightWrist": [0.260, -0.300, 0.132, 0.886],
                "Neck": neutral
            }
        },
        {
            "time": 1.0,
            # Palms together at the chest
            "bones": {
                "LeftShoulder": l_sh,
                "LeftElbow": l_el,
                "LeftWrist": l_wr,
                "RightShoulder": r_sh,
                "RightElbow": r_el,
                "RightWrist": r_wr,
                "Neck": neck_bow
            }
        },
        {
            "time": 1.5,
            # Hold pose and bow slightly deeper
            "bones": {
                "LeftShoulder": l_sh,
                "LeftElbow": l_el,
                "LeftWrist": l_wr,
                "RightShoulder": r_sh,
                "RightElbow": r_el,
                "RightWrist": r_wr,
                "Neck": get_quaternion_from_euler(20, 0, 0)
            }
        },
        {
            "time": 2.0,
            # Return to neutral (hanging down next to torso)
            "bones": {
                "LeftShoulder": l_sh_neutral, "LeftElbow": neutral, "LeftWrist": neutral,
                "RightShoulder": r_sh_neutral, "RightElbow": neutral, "RightWrist": neutral,
                "Neck": neutral
            }
        }
    ]

def generate_hello_keyframes():
    # Hello: Right hand raises up next to head and waves side-to-side.
    neutral = [0.0, 0.0, 0.0, 1.0]
    r_sh_neutral = [0.0, 0.0, -0.7071, 0.7071]
    
    # Palm faces slightly forward-left (tilted 30 degrees towards head)
    palm_tilted = [-0.5, 0.0, 0.866]
    
    sh_raised, el_raised, wr_raised = solve_arm_ik(
        shoulder_pos=[0.4, 0.95, 0.0],
        wrist_target=[0.35, 1.05, 0.25],
        elbow_hint=[0.65, 0.85, 0.15],
        is_left=False,
        hand_dir=[0.0, 1.0, 0.0],
        palm_dir=palm_tilted
    )
    
    sh_w1, el_w1, wr_w1 = solve_arm_ik(
        shoulder_pos=[0.4, 0.95, 0.0],
        wrist_target=[0.42, 1.05, 0.25],
        elbow_hint=[0.70, 0.85, 0.15],
        is_left=False,
        hand_dir=[0.3, 0.95, 0.0],
        palm_dir=palm_tilted
    )
    
    sh_w2, el_w2, wr_w2 = solve_arm_ik(
        shoulder_pos=[0.4, 0.95, 0.0],
        wrist_target=[0.28, 1.05, 0.25],
        elbow_hint=[0.60, 0.85, 0.15],
        is_left=False,
        hand_dir=[-0.3, 0.95, 0.0],
        palm_dir=palm_tilted
    )
    
    return [
        {
            "time": 0.0,
            "bones": {
                "RightShoulder": r_sh_neutral, "RightElbow": neutral, "RightWrist": neutral
            }
        },
        {
            "time": 0.5,
            # Hand raised
            "bones": {
                "RightShoulder": sh_raised,
                "RightElbow": el_raised,
                "RightWrist": wr_raised
            }
        },
        {
            "time": 1.0,
            # Wave right
            "bones": {
                "RightShoulder": sh_w1,
                "RightElbow": el_w1,
                "RightWrist": wr_w1
            }
        },
        {
            "time": 1.5,
            # Wave left
            "bones": {
                "RightShoulder": sh_w2,
                "RightElbow": el_w2,
                "RightWrist": wr_w2
            }
        }
    ]

def generate_thankyou_keyframes():
    # Thank You: Right flat hand starts at lips, then moves forward/down.
    neutral = [0.0, 0.0, 0.0, 1.0]
    r_sh_neutral = [0.0, 0.0, -0.7071, 0.7071]
    
    sh_c, el_c, wr_c = solve_arm_ik(
        shoulder_pos=[0.4, 0.95, 0.0],
        wrist_target=[0.03, 0.88, 0.22],
        elbow_hint=[0.35, 0.65, 0.15],
        is_left=False,
        hand_dir=[-0.3, 0.95, -0.1],
        palm_dir=[-0.2, 0.1, -0.98]
    )
    
    sh_f, el_f, wr_f = solve_arm_ik(
        shoulder_pos=[0.4, 0.95, 0.0],
        wrist_target=[0.20, 0.65, 0.45],
        elbow_hint=[0.45, 0.60, 0.20],
        is_left=False,
        hand_dir=[0.1, 0.9, 0.4],
        palm_dir=[0.0, 0.4, 0.9]
    )
    
    return [
        {
            "time": 0.0,
            "bones": {
                "RightShoulder": r_sh_neutral, "RightElbow": neutral, "RightWrist": neutral
            }
        },
        {
            "time": 0.5,
            # Touch lips/chin
            "bones": {
                "RightShoulder": sh_c,
                "RightElbow": el_c,
                "RightWrist": wr_c
            }
        },
        {
            "time": 1.2,
            # Push hand forward and down
            "bones": {
                "RightShoulder": sh_f,
                "RightElbow": el_f,
                "RightWrist": wr_f
            }
        }
    ]

def generate_yes_keyframes():
    # Yes: Right hand in fist, nodding at the wrist.
    neutral = [0.0, 0.0, 0.0, 1.0]
    r_sh_neutral = [0.0, 0.0, -0.7071, 0.7071]
    
    sh_y, el_y, wr_yd = solve_arm_ik(
        shoulder_pos=[0.4, 0.95, 0.0],
        wrist_target=[0.25, 0.70, 0.30],
        elbow_hint=[0.45, 0.60, 0.15],
        is_left=False,
        hand_dir=[0.0, 0.86, 0.5],
        palm_dir=[-1.0, 0.0, 0.0]
    )
    
    _, _, wr_yu = solve_arm_ik(
        shoulder_pos=[0.4, 0.95, 0.0],
        wrist_target=[0.25, 0.70, 0.30],
        elbow_hint=[0.45, 0.60, 0.15],
        is_left=False,
        hand_dir=[0.0, 0.86, -0.5],
        palm_dir=[-1.0, 0.0, 0.0]
    )
    
    return [
        {
            "time": 0.0,
            "bones": {
                "RightShoulder": r_sh_neutral, "RightElbow": neutral, "RightWrist": neutral
            }
        },
        {
            "time": 0.5,
            # Nod wrist down
            "bones": {
                "RightShoulder": sh_y,
                "RightElbow": el_y,
                "RightWrist": wr_yd
            }
        },
        {
            "time": 1.0,
            # Nod wrist back up
            "bones": {
                "RightShoulder": sh_y,
                "RightElbow": el_y,
                "RightWrist": wr_yu
            }
        },
        {
            "time": 1.5,
            # Nod wrist down again
            "bones": {
                "RightShoulder": sh_y,
                "RightElbow": el_y,
                "RightWrist": wr_yd
            }
        }
    ]

def generate_no_keyframes():
    # No: Right hand index finger pointing up, wagging left-right.
    neutral = [0.0, 0.0, 0.0, 1.0]
    r_sh_neutral = [0.0, 0.0, -0.7071, 0.7071]
    
    sh_n, el_n, wr_nl = solve_arm_ik(
        shoulder_pos=[0.4, 0.95, 0.0],
        wrist_target=[0.25, 0.75, 0.30],
        elbow_hint=[0.45, 0.60, 0.15],
        is_left=False,
        hand_dir=[-0.3, 0.95, 0.0],
        palm_dir=[0.0, 0.0, 1.0]
    )
    
    _, _, wr_nr = solve_arm_ik(
        shoulder_pos=[0.4, 0.95, 0.0],
        wrist_target=[0.25, 0.75, 0.30],
        elbow_hint=[0.45, 0.60, 0.15],
        is_left=False,
        hand_dir=[0.3, 0.95, 0.0],
        palm_dir=[0.0, 0.0, 1.0]
    )
    
    return [
        {
            "time": 0.0,
            "bones": {
                "RightShoulder": r_sh_neutral, "RightElbow": neutral, "RightWrist": neutral
            }
        },
        {
            "time": 0.5,
            # Wag left
            "bones": {
                "RightShoulder": sh_n,
                "RightElbow": el_n,
                "RightWrist": wr_nl
            }
        },
        {
            "time": 1.0,
            # Wag right
            "bones": {
                "RightShoulder": sh_n,
                "RightElbow": el_n,
                "RightWrist": wr_nr
            }
        },
        {
            "time": 1.5,
            # Wag left again
            "bones": {
                "RightShoulder": sh_n,
                "RightElbow": el_n,
                "RightWrist": wr_nl
            }
        }
    ]

def seed_db():
    print("Re-creating all tables in the SQLite database...")
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    
    db = SessionLocal()
    try:
        # Load all video filenames from app/static/videos
        videos_dir = os.path.join(os.path.dirname(__file__), "static", "videos")
        if not os.path.exists(videos_dir):
            # Fallback path if run from elsewhere
            videos_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app", "static", "videos"))
        
        print(f"Loading video files from: {videos_dir}")
        video_files = []
        if os.path.exists(videos_dir):
            video_files = [f[:-4] for f in os.listdir(videos_dir) if f.lower().endswith(".mp4")]
            
        print(f"Found {len(video_files)} signs to seed.")
        
        # Categories mapping
        CATEGORIES_MAP = {
            "colours": ["blue", "black", "brown", "green", "grey", "orange", "pink", "red", "white", "yellow", "colour"],
            "seasons": ["summer", "spring", "winter", "fall", "season", "ex. monsoon", "monsoon"],
            "days and time": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "today", "tomorrow", "yesterday", "week", "month", "year", "time", "morning", "afternoon", "evening", "night", "second", "hour", "minute"],
            "animals": ["animal", "cat", "dog", "cow", "horse", "bird", "fish", "mouse"],
            "clothes": ["clothing", "dress", "hat", "pant", "pocket", "shirt", "shoes", "skirt", "suit", "t-shirt"],
            "pronouns": ["i", "you", "he", "she", "it", "we", "they", "you (plural)"],
            "greetings": ["hello", "good morning", "good afternoon", "good evening", "good night", "thank you", "pleased", "how are you", "alright"],
            "places": ["city", "house", "street or road", "train station", "restaurant", "court", "school", "office", "university", "park", "store or shop", "library", "hospital", "temple", "market", "india", "ground", "bank", "location"],
            "jobs": ["teacher", "student", "lawyer", "doctor", "patient", "waiter", "secretary", "priest", "police", "soldier", "artist", "author", "manager", "reporter", "actor", "job"],
            "transportation": ["plane", "car", "truck", "bicycle", "train", "bus", "boat", "train ticket", "transportation"],
            "home": ["table", "chair", "bed", "dream", "window", "door", "bedroom", "kitchen", "bathroom", "pencil", "pen", "photograph", "soap", "book", "page", "key", "paint", "letter", "paper", "lock", "telephone", "bag", "box", "gift", "card", "ring", "tool"],
            "people": ["son", "daughter", "mother", "father", "parent", "baby", "man", "woman", "brother", "sister", "family", "grandfather", "grandmother", "husband", "wife", "king", "queen", "president", "neighbour", "boy", "girl", "child", "adult", "friend", "player", "crowd"],
            "adjectives": ["loud", "quiet", "happy", "sad", "beautiful", "ugly", "deaf", "blind", "mean", "rich", "poor", "thick", "thin", "expensive", "cheap", "flat", "curved", "male", "female", "tight", "loose", "nice", "high", "low", "soft", "hard", "deep", "shallow", "clean", "dirty", "strong", "weak", "dead", "alive", "heavy", "light", "famous", "long", "short", "tall", "wide", "narrow", "big large", "small little", "slow", "fast", "hot", "cold", "warm", "cool", "new", "bad", "good", "healthy", "sick", "wet", "dry", "old", "young"],
            "society": ["religion", "death", "medicine", "money", "bill", "marriage", "team", "race (ethnicity)", "energy", "war", "peace", "attack", "election", "newspaper", "gun", "technology", "sport", "exercise", "ball", "price", "sign", "science", "god"]
        }
        
        def get_category_for_sign(name):
            for cat, words in CATEGORIES_MAP.items():
                if name.lower() in words:
                    return cat.title()
            return "General"
            
        keyframe_generators = {
            "hello": generate_hello_keyframes,
            "thank you": generate_thankyou_keyframes,
            "yes": generate_yes_keyframes,
            "no": generate_no_keyframes
        }
        
        # Seed Sign Dictionary
        db_signs = []
        for name in video_files:
            cat = get_category_for_sign(name)
            desc = f"Practice the sign for '{name.title()}' in the {cat} category."
            if name == "hello":
                desc = "Raises right hand to the side of the head with the palm facing forward, waving slightly. A friendly opening sign."
            elif name == "thank you":
                desc = "Starts with the flat right hand touching the lips/chin, then moves the hand forward and down toward the listener with palm facing upward."
            elif name == "yes":
                desc = "Forms a fist (S-handshape) with the right hand and tilts it up and down at the wrist, mimicking a nodding head."
            elif name == "no":
                desc = "Moves the right index finger side-to-side in a wagging/shaking motion, expressing negation."
                
            generator = keyframe_generators.get(name.lower())
            anim_data = {"keyframes": generator()} if generator else {"keyframes": []}
            
            sign = SignDictionary(
                sign_name=name.lower(),
                category=cat,
                description=desc,
                animation_data=anim_data
            )
            db.add(sign)
            db_signs.append(sign)
            
        db.commit()
        for s in db_signs:
            db.refresh(s)
            
        print(f"Successfully seeded {len(db_signs)} signs into the dictionary.")
        
        # Group signs by category
        signs_by_cat = {}
        for s in db_signs:
            cat = s.category
            if cat not in signs_by_cat:
                signs_by_cat[cat] = []
            signs_by_cat[cat].append(s)
            
        # Create lessons
        for cat, signs in signs_by_cat.items():
            lesson_title = f"Mastering {cat}"
            lesson_desc = f"Learn and master Indian Sign Language signs related to {cat}."
            signs_sorted = sorted(signs, key=lambda x: x.sign_name)
            
            lesson = Lesson(
                title=lesson_title,
                category=cat,
                difficulty_level="beginner" if cat in ["Greetings", "Pronouns", "Colours"] else "intermediate",
                description=lesson_desc
            )
            db.add(lesson)
            db.commit()
            db.refresh(lesson)
            
            for idx, s in enumerate(signs_sorted):
                item = LessonItem(
                    lesson_id=lesson.id,
                    sign_id=s.id,
                    sort_order=idx + 1
                )
                db.add(item)
            db.commit()
            
        print(f"Successfully seeded {len(signs_by_cat)} lessons and curriculum items dynamically.")
        
    finally:
        db.close()

if __name__ == "__main__":
    seed_db()
