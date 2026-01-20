import cv2
import numpy as np
import mediapipe as mp
import os
import math

# --- Configuration ---
DATASET_PATH = "my_air_dataset"
os.makedirs(DATASET_PATH, exist_ok=True)

# List of characters we want to collect
# You can reduce this list if you only want to test uppercase first
CLASSES = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz' 
current_class_index = 10 # Start at 'A' (Index 10 in the string above)

# ------------------ MediaPipe & Setup ------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)
h_frame, w_frame = 480, 640 # Default usually
canvas = np.zeros((h_frame, w_frame), dtype=np.uint8) 

prev_point = None
action_locked = False
LINE_THICKNESS = 14
MIN_AREA = 500 

# ----------------- Helper Functions -----------------
def check_gesture(lm_list):
    # ... (Keep your exact existing gesture logic here) ...
    # For brevity, I am reusing the logic you provided:
    tx, ty = lm_list[4].x, lm_list[4].y
    ix, iy = lm_list[8].x, lm_list[8].y
    dist_ok = math.hypot(tx - ix, ty - iy)
    ibx, iby = lm_list[5].x, lm_list[5].y
    dist_draw = math.hypot(tx - ibx, ty - iby)

    if dist_ok < 0.07: return "ok"
    
    # Simple finger states
    index_up = lm_list[8].y < lm_list[6].y
    middle_up = lm_list[12].y < lm_list[10].y
    ring_up = lm_list[16].y < lm_list[14].y
    pinky_up = lm_list[20].y < lm_list[18].y

    if not index_up and not middle_up and not ring_up and not pinky_up: return "fist"
    if index_up and middle_up and not ring_up and not pinky_up: return "peace"
    if dist_draw < 0.05: return "draw"
    return "none"

def find_bounding_box(canvas_img):
    # Same as your find_and_merge_boxes
    blur = cv2.GaussianBlur(canvas_img, (3,3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    
    min_x, min_y = 10000, 10000
    max_x, max_y = 0, 0
    total_area = 0
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if w*h < 100: continue
        min_x = min(min_x, x); min_y = min(min_y, y)
        max_x = max(max_x, x + w); max_y = max(max_y, y + h)
        total_area += cv2.contourArea(c)
    
    if total_area < MIN_AREA: return None
    return (min_x, min_y, max_x, max_y)

def save_image(box, canvas_img, char_label):
    x1,y1,x2,y2 = box
    pad = 10
    # Add padding to capture the stroke fully
    x1 = max(0, x1-pad); y1 = max(0, y1-pad)
    x2 = min(canvas_img.shape[1], x2+pad); y2 = min(canvas_img.shape[0], y2+pad)
    
    crop = canvas_img[y1:y2, x1:x2]
    
    # Process exactly like the model expects (Resize to 28x28)
    h, w = crop.shape
    size = max(h, w)
    sq = np.zeros((size, size), dtype=np.uint8)
    y_off = (size - h) // 2
    x_off = (size - w) // 2
    sq[y_off:y_off+h, x_off:x_off+w] = crop
    
    img28 = cv2.resize(sq, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Save logic
    save_dir = os.path.join(DATASET_PATH, char_label)
    os.makedirs(save_dir, exist_ok=True)
    
    count = len(os.listdir(save_dir))
    filename = f"{save_dir}/{count}.png"
    cv2.imwrite(filename, img28)
    print(f"Saved {char_label} sample #{count}")

# ----------------- Main Loop -----------------
print("--- DATA COLLECTION MODE ---")
print("1. Draw a letter.")
print("2. Show 'OK' sign to SAVE it.")
print("3. Press 'n' for Next Class, 'b' for Previous Class.")
print("4. Press 'c' to Clear without saving.")

while True:
    ret, frame = cap.read()
    if not ret: break
    img = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = hands.process(img_rgb)
    h, w = img.shape[:2]
    
    # Update canvas size if needed
    if canvas.shape != (h,w): canvas = np.zeros((h,w), dtype=np.uint8)

    overlay = img.copy()
    current_gesture = "none"
    current_char = CLASSES[current_class_index]

    if res.multi_hand_landmarks:
        lm_list = res.multi_hand_landmarks[0].landmark
        current_gesture = check_gesture(lm_list)
        
        # Drawing Logic
        if current_gesture == "draw":
            ix, iy = int(lm_list[8].x * w), int(lm_list[8].y * h)
            if prev_point is None: prev_point = (ix, iy)
            cv2.line(canvas, prev_point, (ix, iy), 255, LINE_THICKNESS)
            prev_point = (ix, iy)
            action_locked = False
            cv2.circle(img, (ix, iy), 10, (0,255,0), -1)
        else:
            prev_point = None
            
        # Action Logic
        if not action_locked and current_gesture != "draw":
            if current_gesture == "ok":
                box = find_bounding_box(canvas)
                if box:
                    save_image(box, canvas, current_char)
                    # Flash effect
                    cv2.rectangle(overlay, (0,0), (w,h), (0,255,0), 20)
                canvas = np.zeros_like(canvas)
                action_locked = True
            
            elif current_gesture == "fist":
                canvas = np.zeros_like(canvas)
                action_locked = True

    # Display UI
    mask = canvas > 10
    overlay[mask] = (255, 255, 255)
    
    # Info Panel
    cv2.rectangle(overlay, (0,0), (640, 60), (50,50,50), -1)
    cv2.putText(overlay, f"Collecting: {current_char}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    
    # Count how many we have collected for this class
    count_path = os.path.join(DATASET_PATH, current_char)
    if os.path.exists(count_path):
        count = len(os.listdir(count_path))
    else: count = 0
    cv2.putText(overlay, f"Count: {count}", (300, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,200), 2)

    cv2.imshow("Data Collector", overlay)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('n'): # Next letter
        current_class_index = (current_class_index + 1) % len(CLASSES)
    elif key == ord('b'): # Prev letter
        current_class_index = (current_class_index - 1) % len(CLASSES)
    elif key == ord('c'): # Clear
        canvas = np.zeros_like(canvas)

cap.release()
cv2.destroyAllWindows()