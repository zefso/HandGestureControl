import cv2
import mediapipe as mp

class GestureController:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2, 
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

    def run(self):
        cap = cv2.VideoCapture(0)
        print("Система для двох рук активована. 'q' - вихід.")
        
        while cap.isOpened():
            success, image = cap.read()
            if not success: break

            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)

            if results.multi_hand_landmarks:
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    hand_label = results.multi_handedness[i].classification[0].label
                    self.mp_draw.draw_landmarks(
                        image, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    x_coord = int(hand_landmarks.landmark[0].x * image.shape[1])
                    y_coord = int(hand_landmarks.landmark[0].y * image.shape[0])
                    cv2.putText(image, hand_label, (x_coord, y_coord - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow('Double Hand Tracking', image)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    controller = GestureController()
    controller.run()