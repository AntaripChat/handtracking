import numpy as np
import os
import time
import cv2 as cv
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence = 0.8, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret,frame = cap.read()

        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        results = hands.process(image)
        
        image.flags.writeable = True

        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        #print(results)

        if results.multi_hand_landmarks:
            for num,hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)
               


        cv.imshow('Hand Tracking', image)

        if cv.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv.destroyAllWindows()
mp_hands.HAND_CONNECTIONS



