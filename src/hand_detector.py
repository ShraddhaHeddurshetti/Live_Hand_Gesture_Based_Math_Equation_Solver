import mediapipe as mp
import cv2
import numpy as np

class HandDetector:
    def __init__(self, min_detection_confidence=0.7, min_tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            max_num_hands=1  # Limit to one hand for better performance
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Add gesture stabilization
        self.last_gesture = None
        self.gesture_counter = 0
        self.GESTURE_STABILITY_FRAMES = 3
        
    def find_hands(self, img, draw=True):
        """Detect hands in an image and optionally draw landmarks"""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(
                        img, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS
                    )
        return img
    
    def get_gesture(self):
        """Determine the current gesture based on hand landmarks with stabilization"""
        if not hasattr(self, "results") or not self.results.multi_hand_landmarks:
            self.last_gesture = None
            self.gesture_counter = 0
            return None
            
        # Get landmarks of the first hand
        landmarks = self.results.multi_hand_landmarks[0].landmark
        
        # Calculate finger states (up/down)
        fingers = self._count_fingers(landmarks)
        
        # Determine current gesture
        current_gesture = None
        
        if sum(fingers) == 1 and fingers[1]:  
            current_gesture = "WRITE"       # Only index finger up
        elif sum(fingers) == 2 and fingers[1] and fingers[2]:  
            current_gesture = "ERASE"       # Index + middle fingers up
        elif sum(fingers) >= 4:  
            current_gesture = "CLEAR"       # All/most fingers up
        elif sum(fingers) == 0:  
            current_gesture = "SOLVE"       # Fist
        elif sum(fingers) == 3 and fingers[1] and fingers[2] and fingers[3]:  
            current_gesture = "HOVER"       # Three middle fingers up
        
        # Gesture stabilization
        if current_gesture == self.last_gesture:
            self.gesture_counter += 1
        else:
            self.gesture_counter = 0
            
        self.last_gesture = current_gesture
        
        if self.gesture_counter >= self.GESTURE_STABILITY_FRAMES:
            return current_gesture
            
        return None
        
    def _count_fingers(self, landmarks):
        """Count which fingers are up with improved tolerance"""
        fingers = []
        
        # Thumb
        thumb_tip = landmarks[4]
        thumb_base = landmarks[2]
        if thumb_tip.x < thumb_base.x - 0.02:  
            fingers.append(1)
        else:
            fingers.append(0)
            
        # Other 4 fingers
        tips = [8, 12, 16, 20]
        mids = [6, 10, 14, 18]
        
        for tip, mid in zip(tips, mids):
            if landmarks[tip].y < landmarks[mid].y - 0.03:
                fingers.append(1)
            else:
                fingers.append(0)
                
        return fingers
