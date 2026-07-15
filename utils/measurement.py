import cv2
import numpy as np


def measure_nails(mask_polygons, pixels_per_mm):
    """
    mask_polygons : YOLO segmentation polygons
    pixels_per_mm : Coin detector se mila pixels/mm ratio
    """

    measurements = []

    for polygon in mask_polygons:

        polygon = polygon.astype(np.int32)

        # Bounding box
        x, y, w, h = cv2.boundingRect(polygon)

        # Mask create
        mask = np.zeros((h + 10, w + 10), dtype=np.uint8)

        shifted = polygon - [x, y]

        cv2.fillPoly(mask, [shifted], 255)

        max_width = 0
        max_y = 0

        # Maximum nail width find karo
        for row in range(mask.shape[0]):

            cols = np.where(mask[row] == 255)[0]

            if len(cols) > 1:

                width = cols[-1] - cols[0]

                if width > max_width:
                    max_width = width
                    max_y = row

        width_px = float(max_width)
        height_px = float(h)

        # ⭐⭐⭐ IMPORTANT ⭐⭐⭐
        # Coin detector se aaye pixels/mm ratio ka use karo
        width_mm = round(width_px / pixels_per_mm, 2)
        height_mm = round(height_px / pixels_per_mm, 2)

        print("----------------------------")
        print("Width(px):", width_px)
        print("Height(px):", height_px)
        print("Pixels/mm:", pixels_per_mm)
        print("Width(mm):", width_mm)
        print("Height(mm):", height_mm)
        print("----------------------------")

        measurements.append({
            "width_px": round(width_px, 2),
            "height_px": round(height_px, 2),
            "width_mm": width_mm,
            "height_mm": height_mm,
            "center_x": x + w / 2,
            "center_y": y + h / 2,
            "polygon": polygon,
            "measure_row": max_y + y
        })

    # Left → Right order
    measurements.sort(key=lambda m: m["center_x"])

    return measurements
