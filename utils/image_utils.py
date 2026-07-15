import cv2
import numpy as np

def draw_measurements(image_path, mask_polygons, measurements, output_path):

    image = cv2.imread(image_path)

    for i, (polygon, nail) in enumerate(zip(mask_polygons, measurements), start=1):

        polygon = polygon.astype(np.int32)

        # Green outline
        cv2.polylines(
            image,
            [polygon],
            True,
            (0,255,0),
            2
        )

        # Center
        cx = int(nail["center_x"])
        cy = int(nail["center_y"])

        text = f"N{i}: {nail['width_px']} px"

        cv2.putText(
            image,
            text,
            (cx-30, cy-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,0,255),
            2
        )

    cv2.imwrite(output_path, image)

    return output_path