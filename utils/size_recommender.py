# utils/size_recommender.py

SIZE_CHART = [
    {"size": "0", "min": 17.0, "max": 18.5},
    {"size": "1", "min": 16.0, "max": 17.0},
    {"size": "2", "min": 15.0, "max": 16.0},
    {"size": "3", "min": 14.0, "max": 15.0},
    {"size": "4", "min": 13.0, "max": 14.0},
    {"size": "5", "min": 12.0, "max": 13.0},
    {"size": "6", "min": 11.0, "max": 12.0},
    {"size": "7", "min": 10.0, "max": 11.0},
    {"size": "8", "min": 9.0, "max": 10.0},
    {"size": "9", "min": 0.0, "max": 9.0},
]


def recommend_size(width_mm):
    if width_mm is None:
        return "Unknown"

    try:
        width_mm = float(width_mm)
    except ValueError:
        return "Unknown"

    # 🔴 FIX: Agar width chart ki sabse badi limit (18.5 mm) se zyada hai,
    # toh usse automatic sabse bada size यानी "0" de do.
    if width_mm >= 18.5:
        return "0"

    for item in SIZE_CHART:
        if item["min"] <= width_mm < item["max"]:
            return item["size"]

    return "Unknown"