import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


def classify_posture(landmarks):
    if not landmarks:
        return "No Detection"
    
    left_hand_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
    right_hand_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
    left_elbow_y = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y
    right_elbow_y = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y

    if left_hand_y < left_elbow_y and right_hand_y < right_elbow_y:
        return "stop"
    elif left_hand_y < left_elbow_y and right_hand_y > right_elbow_y:
        return "turn_right"
    elif right_hand_y < right_elbow_y and left_hand_y > left_elbow_y:
        return "turn_left"
    elif left_hand_y > left_elbow_y and right_hand_y > right_elbow_y:
        return "advance"
    else:
        return "unknown"


class SignDetector(Node):
    def __init__(self):
        super().__init__('sign_detector')
        self.publisher_ = self.create_publisher(String, '/atc/orders', 10)
        self.timer = self.create_timer(0.1, self.detect_gesture)
        self.cap = cv2.VideoCapture(0)
        self.get_logger().info("SignDetector Node Started")

    def detect_gesture(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("Failed to capture frame")
            return

        frame = cv2.resize(frame, (640, 480))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        landmarks = results.pose_landmarks.landmark if results.pose_landmarks else None
        instruction = classify_posture(landmarks)

        msg = String()
        msg.data = instruction
        self.publisher_.publish(msg)
        self.get_logger().info(f"Published Instruction: {instruction}")

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.putText(
            frame,
            f"Instruction: {instruction}",
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Traffic Controller Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.cap.release()
            cv2.destroyAllWindows()
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    sign_detector = SignDetector()
    rclpy.spin(sign_detector)
    sign_detector.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
