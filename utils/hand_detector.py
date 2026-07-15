import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def detect_hand(image_path, output_path):

    image = cv2.imread(image_path)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    ) as hands:

        results = hands.process(rgb)

        landmarks = []

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                mp_draw.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                h, w, _ = image.shape

                for lm in hand_landmarks.landmark:

                    landmarks.append((
                        int(lm.x * w),
                        int(lm.y * h)
                    ))

        cv2.imwrite(output_path, image)

    return landmarks