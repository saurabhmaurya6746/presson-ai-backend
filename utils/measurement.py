import cv2
import numpy as np


def pixel_to_mm(pixel, coin_pixels):
    COIN_MM = 27.0
    scale = COIN_MM / coin_pixels
    return round(pixel * scale, 2)


def measure_nails(mask_polygons):
    measurements = []
    
    # TODO: Yahan aap apni coin_diameter ki pixel value nikal lo 
    # (Example ke liye maine 100.0 rakh diya hai)
    coin_diameter = 100.0 

    for polygon in mask_polygons:
        polygon = polygon.astype(np.int32)

        # Mask banana
        x, y, w, h = cv2.boundingRect(polygon)
        mask = np.zeros((h + 10, w + 10), dtype=np.uint8)
        shifted = polygon - [x, y]
        cv2.fillPoly(mask, [shifted], 255)

        max_width = 0
        max_y = 0

        # Har row check karo
        for row in range(mask.shape[0]):
            cols = np.where(mask[row] == 255)[0]
            if len(cols) > 1:
                width = cols[-1] - cols[0]
                if width > max_width:
                    max_width = width
                    max_y = row

        # Step 2: width aur height variables nikalna
        width_px = round(float(max_width), 2)
        height_px = round(float(h), 2)
        print(width_px)
        print(height_px)
        # mm me convert karna
        width_mm = pixel_to_mm(width_px, coin_diameter)
        height_mm = pixel_to_mm(height_px, coin_diameter)
        print(width_mm)
        print(height_mm)
        # Dictionary me add karna jo aapne manga tha
        measurements.append({
            "width_px": width_px,
            "height_px": height_px,
            "width_mm": width_mm,
            "height_mm": height_mm,
            "center_x": x + w / 2,
            "center_y": y + h / 2,
            "polygon": polygon,
            "measure_row": max_y + y
        })

    measurements.sort(key=lambda m: m["center_x"])

    return measurements