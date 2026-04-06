import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import math

# إعداد الصوت و MediaPipe
engine = pyttsx3.init()
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(p1, p2, p3):
    a = np.array([p1.x, p1.y])
    b = np.array([p2.x, p2.y])
    c = np.array([p3.x, p3.y])
    radians = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

def get_letter(hand_landmarks):
    lm = hand_landmarks.landmark
    
    # حساب زوايا الأصابع الرئيسية
    ang_i = calculate_angle(lm[5], lm[6], lm[8])   # السبابة
    ang_m = calculate_angle(lm[9], lm[10], lm[12]) # الوسطى
    ang_r = calculate_angle(lm[13], lm[14], lm[16])# البنصر
    ang_p = calculate_angle(lm[17], lm[18], lm[20])# الخنصر

    # حالة الأصابع (مفتوح/مغلق)
    is_i = ang_i > 150
    is_m = ang_m > 150
    is_r = ang_r > 150
    is_p = ang_p > 150

    # مسافات للتمييز الدقيق
    d_ti = np.linalg.norm(np.array([lm[4].x, lm[4].y]) - np.array([lm[8].x, lm[8].y]))
    d_im = np.linalg.norm(np.array([lm[8].x, lm[8].y]) - np.array([lm[12].x, lm[12].y]))
    d_th_m = np.linalg.norm(np.array([lm[4].x, lm[4].y]) - np.array([lm[10].x, lm[10].y]))

    # --- منطق الحروف الكامل A-Z ---
    
    # 1. الحروف المعتمدة على الأصابع المرفوعة
    if is_i and is_m and is_r and is_p: return "B"
    if is_i and not is_m and not is_r and not is_p: return "D"
    if is_p and not is_i and not is_m and not is_r: return "I"
    if is_i and is_m and not is_r and not is_p:
        return "V" if d_im > 0.06 else "U"
    if is_i and is_m and is_r and not is_p: return "W"
    if is_i and lm[4].x < lm[2].x and not is_m: return "L"
    if is_p and lm[4].x < lm[2].x and not is_i: return "Y"
    if not is_i and is_m and is_r and is_p and d_ti < 0.05: return "F"

    # 2. الحروف المعتمدة على القبضة (Fist) وحركة الإبهام
    if not is_i and not is_m and not is_r and not is_p:
        if lm[4].y < lm[5].y and lm[4].x > lm[5].x: return "A" # الإبهام للخارج
        if d_ti < 0.05 and d_th_m < 0.05: return "O"           # شكل دائرة
        if 70 < ang_i < 120: return "C"                       # شكل هلال
        if lm[4].y > lm[8].y and lm[4].y > lm[12].y: return "E" # قبضة محكمة
        if lm[4].x < lm[13].x: return "M"                     # إبهام تحت 3 أصابع
        if lm[4].x < lm[9].x: return "N"                      # إبهام تحت إصبعين
        return "S"                                            # إبهام فوق الأصابع

    # 3. حالات خاصة (G, H, K, P, X)
    if is_i and not is_m and abs(lm[8].y - lm[5].y) < 0.05: return "G"
    if is_i and is_m and abs(lm[12].y - lm[9].y) < 0.05: return "H"
    if is_i and is_m and d_ti < 0.06: return "P" if lm[8].y > lm[5].y else "K"
    if 50 < ang_i < 100 and not is_m: return "X"

    return ""

# الحلقة الرئيسية
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    if results.multi_hand_landmarks:
        for res in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, res, mp_hands.HAND_CONNECTIONS)
            char = get_letter(res)
            cv2.putText(frame, char, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 4)

    cv2.imshow('ASL Full Detector', frame)
    if cv2.waitKey(1) == ord('q'): break
cap.release()
cv2.destroyAllWindows()
