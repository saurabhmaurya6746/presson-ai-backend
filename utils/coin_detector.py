# utils/coin_detector.py
import cv2
import numpy as np


def get_pixel_to_mm_ratio(image_path, real_coin_diameter_mm=25.0):
    """Coin detect karke uska ratio aur drawing coordinates return karta hai."""
    image = cv2.imread(image_path)
    if image is None:
        return 4.5, None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=100,
        param2=30,
        minRadius=30,
        maxRadius=200,
    )

    if circles is None:
        # Fallback: ratio = 4.5, coin_data = None
        return 4.5, None

    circles = np.uint16(np.around(circles))
    x, y, r = circles[0][0]
    diameter_px = int(r * 2)

    # Backend verification ke liye console me print
    print("\n" + "=" * 40)
    print(f"[BACKEND INFO] Coin Detected!")
    print(f"-> Center: ({x}, {y}), Radius: {r}px")
    print(f"-> Diameter in Pixels: {diameter_px}px")
    print(f"-> Calculated Ratio (px/mm): {diameter_px / real_coin_diameter_mm:.4f}")
    print("=" * 40 + "\n")

    pixels_per_mm = diameter_px / real_coin_diameter_mm
    coin_data = {"center": (int(x), int(y)), "radius": int(r), "diameter_px": diameter_px}

    return pixels_per_mm, coin_data