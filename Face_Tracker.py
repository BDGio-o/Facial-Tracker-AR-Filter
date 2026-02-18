import cv2
import mediapipe as mp
import numpy as np

# Pinch detection
def is_pinch(hand_landmarks, frame_w, frame_h, threshold=25):
    thumb_tip = hand_landmarks.landmark[4]
    thumb_joint = hand_landmarks.landmark[8]
    index_tip = hand_landmarks.landmark[8]
    index_joint = hand_landmarks.landmark[6]

    tx, ty = int(thumb_tip.x * frame_w), int(thumb_tip.y * frame_h)
    ix, iy = int(index_tip.x * frame_w), int(index_tip.y * frame_h)

    tip_distance = np.hypot(ix - tx, iy - ty)

    thumb_bent = thumb_tip.y > thumb_joint.y
    index_bent = index_tip.y > index_joint.y

    return tip_distance < threshold and thumb_bent and index_bent, (ix, iy)

# Overlay PNG with alpha channel
def overlay_image_alpha(img, img_overlay, x, y, overlay_size=None):
    if overlay_size is not None:
        img_overlay = cv2.resize(img_overlay, overlay_size)

    b, g, r, a = cv2.split(img_overlay)
    overlay_color = cv2.merge((b, g, r))
    mask = cv2.merge((a, a, a))

    h, w, _ = overlay_color.shape

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(img.shape[1], x + w), min(img.shape[0], y + h)

    if x1 >= x2 or y1 >= y2:
        return

    roi = img[y1:y2, x1:x2]
    mask_roi = mask[0:(y2 - y1), 0:(x2 - x1)]
    overlay_roi = overlay_color[0:(y2 - y1), 0:(x2 - x1)]

    img[y1:y2, x1:x2] = cv2.add(
        cv2.bitwise_and(roi, 255 - mask_roi),
        cv2.bitwise_and(overlay_roi, mask_roi)
    )

# Loading assets/Pngs
hat = cv2.imread("hat.png", cv2.IMREAD_UNCHANGED)
moustache = cv2.imread("moustache.png", cv2.IMREAD_UNCHANGED)
glasses = cv2.imread("glasses.png", cv2.IMREAD_UNCHANGED)

if hat is None or moustache is None or glasses is None:
    print("Error loading accessory images")
    exit()

glasses_width = 200
glasses_height = int(glasses_width * glasses.shape[0] / glasses.shape[1])

#State Variables
glasses_x, glasses_y = None, None
glasses_grabbed = False
active_hand_id = None
glasses_on_head = True
glasses_offset_x = 0
glasses_offset_y = 0

show_boxes = True
show_ar = True
show_hat = False
show_moustache = False


#mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

green_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)

FEATURES = {
    "Left Eye": [33, 133],
    "Right Eye": [362, 263],
    "Mouth": [61, 291],
    "Nose": [1, 2],
}

PADDING = 10

#accessing webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to close program, 'b' to toggle boxes ON/OFF")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #face detection
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        #drawing box around head
        if show_boxes:
            all_x = [int(landmark.x * w) for landmark in face_landmarks.landmark]
            all_y = [int(landmark.y * h) for landmark in face_landmarks.landmark]

            head_x_min = max(min(all_x) - PADDING, 0)
            head_x_max = min(max(all_x) + PADDING, w)
            head_y_min = max(min(all_y) - PADDING, 0)
            head_y_max = min(max(all_y) + PADDING, h)

            cv2.rectangle(frame,
                          (head_x_min, head_y_min),
                          (head_x_max, head_y_max),
                          (0, 255, 0), 2)

            cv2.putText(frame,
                        "Head",
                        (head_x_min, head_y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA)

        #drawing boxes around landmarks
        if show_boxes:
            for feature_name, indices in FEATURES.items():
                x_coords = [int(face_landmarks.landmark[i].x * w) for i in indices]
                y_coords = [int(face_landmarks.landmark[i].y * h) for i in indices]

                x_min, x_max = max(min(x_coords) - PADDING, 0), min(max(x_coords) + PADDING, w)
                y_min, y_max = max(min(y_coords) - PADDING, 0), min(max(y_coords) + PADDING, h)

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, feature_name, (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        #face landmarks
        left_eye = face_landmarks.landmark[33]
        forehead = face_landmarks.landmark[6]

        head_x = int(left_eye.x * w)
        head_y = int(forehead.y * h)

        if glasses_x is None:
            glasses_x = head_x
            glasses_y = head_y

        #Anchor for accsesories
        forehead_top = face_landmarks.landmark[10]
        nose_tip = face_landmarks.landmark[1]
        upper_lip = face_landmarks.landmark[13]
        left_cheek = face_landmarks.landmark[234]
        right_cheek = face_landmarks.landmark[454]

        fx, fy = int(forehead_top.x * w), int(forehead_top.y * h)
        nx, ny = int(nose_tip.x * w), int(nose_tip.y * h)
        ulx, uly = int(upper_lip.x * w), int(upper_lip.y * h)
        lx, _ = int(left_cheek.x * w), int(left_cheek.y * h)
        rx, _ = int(right_cheek.x * w), int(right_cheek.y * h)

        face_width = abs(rx - lx)

        #Making hat appear
        if show_hat:
            hat_width = int(face_width * 1.2)
            hat_height = int(hat.shape[0] * (hat_width / hat.shape[1]))
            hat_x = fx - hat_width // 2
            hat_y = fy - hat_height + 20
            overlay_image_alpha(frame, hat, hat_x, hat_y, (hat_width, hat_height))

        #Making moustache appear
        if show_moustache:
            moustache_width = int(face_width * 1.2)
            moustache_height = int(moustache.shape[0] * (moustache_width / moustache.shape[1]))
            moustache_x = nx - moustache_width // 2
            moustache_y = uly - moustache_height // 2
            overlay_image_alpha(frame, moustache, moustache_x, moustache_y,
                                (moustache_width, moustache_height))

    #hand detection and glasses interaction
    hand_results = hands.process(rgb_frame)

    if hand_results.multi_hand_landmarks:
        for hand_id, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=green_drawing_spec,
                connection_drawing_spec=green_drawing_spec
            )

            pinching, pinch_point = is_pinch(hand_landmarks, w, h)

            if pinching and not glasses_grabbed:
                dx = pinch_point[0] - (glasses_x + glasses_width // 2)
                dy = pinch_point[1] - (glasses_y + glasses_height // 2)
                if np.hypot(dx, dy) < 120:
                    glasses_grabbed = True
                    active_hand_id = hand_id
                    glasses_on_head = False

            if glasses_grabbed and active_hand_id == hand_id:
                if pinching:
                    glasses_x = pinch_point[0] - glasses_width // 2
                    glasses_y = pinch_point[1] - glasses_height // 2
                else:
                    dx = glasses_x - head_x
                    dy = glasses_y - head_y

                    if np.hypot(dx, dy) < 100:
                        glasses_on_head = True
                        glasses_offset_x = dx
                        glasses_offset_y = dy
                    else:
                        glasses_on_head = False
                        glasses_offset_x = 0
                        glasses_offset_y = 0

                    glasses_grabbed = False
                    active_hand_id = None

    #glasses following head
    if glasses_on_head and results.multi_face_landmarks:
        glasses_x = head_x + glasses_offset_x
        glasses_y = head_y + glasses_offset_y

    #making glasses appear
    if show_ar and glasses_x is not None:
        overlay_image_alpha(frame, glasses, glasses_x, glasses_y,
                            (glasses_width, glasses_height))

    #display and input
    cv2.imshow("Face Tracker", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('b'):
        show_boxes = not show_boxes
    elif key == ord('g'):
        show_ar = not show_ar
    elif key == ord('h'):
        show_hat = not show_hat
    elif key == ord('m'):
        show_moustache = not show_moustache


cap.release()
cv2.destroyAllWindows()
