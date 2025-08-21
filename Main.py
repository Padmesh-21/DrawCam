import cv2
import numpy as np
import mediapipe as mp
import pygame

pygame.init()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1,min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
_, frame = cap.read()
height, width, _ = frame.shape


screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Virtual Drawing with Finger")


canvas = np.zeros((height, width, 3), dtype=np.uint8)


prev_x, prev_y = None, None
color = (0, 255, 255)  
thickness = 5

def is_index_open(hand_landmarks):
    return hand_landmarks.landmark[8].y < hand_landmarks.landmark[5].y

def is_hand_closed(hand_landmarks):
    finger_tips = [8, 12, 16, 20]
    finger_joints = [5, 10, 14, 18]

    closed_fingers = 0

    for tip, joint in zip(finger_tips, finger_joints):
        if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[joint].y:
            closed_fingers += 1
    return closed_fingers == 3

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    ret, frame = cap.read()
    if not ret:
        break
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        hand_landmarks  = result.multi_hand_landmarks[0]
        x, y = int(hand_landmarks.landmark[8].x * width), int(hand_landmarks.landmark[8].y * height)
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        if prev_x is not None and prev_y is not None and is_hand_closed(hand_landmarks) and is_index_open(hand_landmarks):
            cv2.line(canvas, (prev_x, prev_y), (x, y), color, thickness)
        prev_x, prev_y = x, y
        if is_index_open(hand_landmarks) and y < 50 and  x < 50 :
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            pass
    else:
        prev_x, prev_y = None, None 

    frame = cv2.addWeighted(frame, 0.7, canvas, 0.7, 0)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    surface = pygame.surfarray.make_surface(np.rot90(frame))

    screen.blit(surface, (0, 0))
    pygame.display.update()

    keys = pygame.key.get_pressed()
    if keys[pygame.K_c]:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

cap.release()
pygame.quit()