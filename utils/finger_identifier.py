import math
# 🔴 Naya import size recommend karne ke liye
from utils.size_recommender import recommend_size


def identify_fingers(hand_landmarks, nail_measurements):
    # MediaPipe ke standard landmark indices standard format me
    # hand_landmarks ek list honi chahiye jisme har landmark ke paas x aur y coordinates honge
    FINGER_TIPS = {
        "Thumb": 4,
        "Index": 8,
        "Middle": 12,
        "Ring": 16,
        "Little": 20,
    }

    # Agar landmarks ya nail measurements missing hain toh empty return karein
    if not hand_landmarks or not nail_measurements:
        return []

    result = []

    # Ek tracker banayenge taaki ek hi nail do fingers ko assign na ho jaye
    assigned_nails = set()

    for finger_name, tip_idx in FINGER_TIPS.items():
        # Edge case handling: check karein ki landmark list me wo index exist karta hai
        if tip_idx >= len(hand_landmarks):
            continue

        tip = hand_landmarks[tip_idx]

        # MediaPipe coordinates normalized hote hain (0 se 1).
        # Agar aapki hand_landmarks me pixels hain toh direct use hoga.
        tip_x = tip.x if hasattr(tip, "x") else tip[0]
        tip_y = tip.y if hasattr(tip, "y") else tip[1]

        min_distance = float("inf")
        closest_nail = None
        closest_nail_idx = -1

        for idx, nail in enumerate(nail_measurements):
            # Agar yeh nail pehle hi kisi finger ko mil chuka hai toh skip karein
            if idx in assigned_nails:
                continue

            nail_cx = nail["center_x"]
            nail_cy = nail["center_y"]

            # Euclidean distance formula: sqrt((x1-x2)^2 + (y1-y2)^2)
            distance = math.sqrt(
                (tip_x - nail_cx) ** 2 + (tip_y - nail_cy) ** 2
            )

            if distance < min_distance:
                min_distance = distance
                closest_nail = nail
                closest_nail_idx = idx

        if closest_nail:
            # Nail ko block kar do taaki koi aur finger isse na le sake
            assigned_nails.add(closest_nail_idx)

            # 🔴 Size calculation yahan generate ho raha hai aur direct dictionary me append ho raha hai
            recommended_size_val = recommend_size(closest_nail["width_mm"])

            result.append({
                "finger": finger_name,
                "width_mm": closest_nail["width_mm"],
                "height_mm": closest_nail["height_mm"],
                "width_px": closest_nail["width_px"],
                "height_px": closest_nail["height_px"],
                "size": recommended_size_val,  # 🔴 Naya key-value pair map ho gaya
            })

    return result