import cv2
import numpy as np
from collections import deque
import time
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    from deepface import DeepFace
    print(" DeepFace imported successfully")
except ImportError:
    print(" DeepFace not found. Run: pip install deepface tf-keras")
    exit(1)

class EmotionDetector:
    def __init__(self):
        print(" Initializing Emotion Detection System...")
        
        # Load Haar Cascade for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Load eye cascade for blink detection
        eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
        self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
        
        if self.face_cascade.empty():
            print(" Error loading face cascade!")
            exit(1)
        
        # Emotion colors (BGR)
        self.emotion_colors = {
            'happy': (50, 205, 50),      # Lime Green
            'sad': (255, 100, 100),      # Light Red
            'angry': (0, 0, 255),        # Red
            'surprise': (0, 255, 255),   # Yellow
            'fear': (180, 105, 255),     # Pink
            'disgust': (0, 128, 128),    # Teal
            'neutral': (200, 200, 200)   # Gray
        }
        
        # Emotion icons
        self.emotion_icons = {
            'happy': '😊',
            'sad': '😢',
            'angry': '😠',
            'surprise': '😮',
            'fear': '😨',
            'disgust': '🤢',
            'neutral': '😐'
        }
        
        # State tracking
        self.current_emotion = 'neutral'
        self.emotion_scores = {}
        self.fps = 0
        self.frame_times = deque(maxlen=30)
        self.blink_counter = 0
        self.last_blink_time = time.time()
        
        # DeepFace model warming up
        print(" Warming up AI model...")
        try:
            dummy = np.zeros((100, 100, 3), dtype=np.uint8)
            DeepFace.analyze(dummy, actions=['emotion'], enforce_detection=False, silent=True)
            print("AI Model ready!")
        except:
            pass
        
        print(" System initialized successfully!")
    
    def detect_emotion(self, face_img):
        """Detect emotion with error handling"""
        try:
            result = DeepFace.analyze(
                face_img, 
                actions=['emotion'], 
                enforce_detection=False,
                silent=True
            )
            
            if isinstance(result, list):
                result = result[0]
            
            self.current_emotion = result['dominant_emotion']
            self.emotion_scores = result['emotion']
            return True
            
        except Exception as e:
            return False
    
    def detect_gestures(self, face_region):
        """Detect eye blinks"""
        gestures = []
        
        # Detect eyes for blink detection
        eyes = self.eye_cascade.detectMultiScale(face_region, 1.1, 5, minSize=(20, 20))
        
        # Blink detection
        if len(eyes) < 2:
            current_time = time.time()
            if current_time - self.last_blink_time > 0.3:  # Debounce
                self.blink_counter += 1
                self.last_blink_time = current_time
                gestures.append("Blink")
        
        return gestures, len(eyes)
    
    def draw_emotion_bar(self, frame, emotion, score, y_pos, panel_x):
        """Draw emotion probability bar with proper alignment"""
        bar_width = 180
        bar_height = 22
        x_start = panel_x + 20
        
        bar_fill = int(score * bar_width)
        color = self.emotion_colors.get(emotion, (255, 255, 255))
        
        # Background bar
        cv2.rectangle(frame, (x_start, y_pos), (x_start + bar_width, y_pos + bar_height), 
                     (40, 40, 40), -1)
        
        # Filled bar with gradient effect
        if bar_fill > 0:
            cv2.rectangle(frame, (x_start, y_pos), (x_start + bar_fill, y_pos + bar_height), 
                         color, -1)
        
        # Border
        cv2.rectangle(frame, (x_start, y_pos), (x_start + bar_width, y_pos + bar_height), 
                     (80, 80, 80), 1)
        
        # Emotion label (left aligned)
        label = emotion.capitalize()
        cv2.putText(frame, label, (panel_x + 5, y_pos - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        
        # Percentage (right aligned on bar)
        percent_text = f"{score:.1%}"
        cv2.putText(frame, percent_text, (x_start + bar_width - 50, y_pos + 16),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    
    def draw_face_box(self, frame, x, y, w, h):
        """Draw clean face detection box"""
        color = self.emotion_colors.get(self.current_emotion, (0, 255, 0))
        thickness = 3
        
        # Main rectangle with rounded corners effect
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        
        # Corner decorations (longer and thicker)
        corner_len = 30
        corner_thick = 4
        
        # Top-left
        cv2.line(frame, (x, y), (x + corner_len, y), color, corner_thick)
        cv2.line(frame, (x, y), (x, y + corner_len), color, corner_thick)
        
        # Top-right
        cv2.line(frame, (x + w, y), (x + w - corner_len, y), color, corner_thick)
        cv2.line(frame, (x + w, y), (x + w, y + corner_len), color, corner_thick)
        
        # Bottom-left
        cv2.line(frame, (x, y + h), (x + corner_len, y + h), color, corner_thick)
        cv2.line(frame, (x, y + h), (x, y + h - corner_len), color, corner_thick)
        
        # Bottom-right
        cv2.line(frame, (x + w, y + h), (x + w - corner_len, y + h), color, corner_thick)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_len), color, corner_thick)
        
        # Label with icon at top
        icon = self.emotion_icons.get(self.current_emotion, '')
        label_text = f"{self.current_emotion.upper()} {icon}"
        
        # Get text size for background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thick = 2
        (text_width, text_height), _ = cv2.getTextSize(label_text, font, font_scale, font_thick)
        
        # Draw label background
        label_y = y - 10
        cv2.rectangle(frame, (x, label_y - text_height - 10), 
                     (x + text_width + 15, label_y + 5), color, -1)
        
        # Draw label text
        cv2.putText(frame, label_text, (x + 7, label_y - 5),
                   font, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)
    
    def draw_hud(self, frame, face_detected, gestures, eye_count):
        """Draw professional HUD with proper layout"""
        h, w = frame.shape[:2]
        
        # ===== TOP BAR =====
        top_bar_height = 70
        cv2.rectangle(frame, (0, 0), (w, top_bar_height), (25, 25, 25), -1)
        cv2.rectangle(frame, (0, 0), (w, top_bar_height), (0, 200, 255), 3)
        
        # Title (left)
        cv2.putText(frame, "EMOTION DETECTION SYSTEM", (25, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3, cv2.LINE_AA)
        
        # AI Powered badge (right)
        cv2.putText(frame, "AI POWERED", (w - 200, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
        
        # ===== LEFT PANEL - DETECTION INFO =====
        left_panel_x = 20
        left_panel_y = top_bar_height + 20
        left_panel_w = 280
        left_panel_h = 160
        
        # Panel background
        cv2.rectangle(frame, (left_panel_x, left_panel_y), 
                     (left_panel_x + left_panel_w, left_panel_y + left_panel_h), 
                     (25, 25, 25), -1)
        cv2.rectangle(frame, (left_panel_x, left_panel_y), 
                     (left_panel_x + left_panel_w, left_panel_y + left_panel_h), 
                     (0, 255, 0), 2)
        
        # Panel header
        cv2.rectangle(frame, (left_panel_x, left_panel_y), 
                     (left_panel_x + left_panel_w, left_panel_y + 35), 
                     (0, 200, 0), -1)
        cv2.putText(frame, "DETECTION INFO", (left_panel_x + 45, left_panel_y + 23),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Panel content
        content_y = left_panel_y + 60
        line_height = 35
        
        cv2.putText(frame, f"Eyes Detected: {eye_count}", 
                   (left_panel_x + 20, content_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        
        cv2.putText(frame, f"Total Blinks: {self.blink_counter}", 
                   (left_panel_x + 20, content_y + line_height),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        
        if gestures:
            cv2.putText(frame, f"Status: {gestures[0]}", 
                       (left_panel_x + 20, content_y + line_height * 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        
        # ===== RIGHT PANEL - EMOTION ANALYSIS =====
        right_panel_w = 280
        right_panel_x = w - right_panel_w - 20
        right_panel_y = top_bar_height + 20
        right_panel_h = 340
        
        if self.emotion_scores:
            # Panel background
            cv2.rectangle(frame, (right_panel_x, right_panel_y), 
                         (right_panel_x + right_panel_w, right_panel_y + right_panel_h), 
                         (25, 25, 25), -1)
            cv2.rectangle(frame, (right_panel_x, right_panel_y), 
                         (right_panel_x + right_panel_w, right_panel_y + right_panel_h), 
                         (0, 200, 255), 2)
            
            # Panel header
            cv2.rectangle(frame, (right_panel_x, right_panel_y), 
                         (right_panel_x + right_panel_w, right_panel_y + 35), 
                         (0, 150, 255), -1)
            cv2.putText(frame, "EMOTION ANALYSIS", (right_panel_x + 40, right_panel_y + 23),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Emotion bars - properly spaced
            sorted_emotions = sorted(self.emotion_scores.items(), 
                                   key=lambda x: x[1], reverse=True)
            
            bar_start_y = right_panel_y + 55
            bar_spacing = 42
            
            for idx, (emotion, score) in enumerate(sorted_emotions):
                y_pos = bar_start_y + idx * bar_spacing
                self.draw_emotion_bar(frame, emotion, score / 100, y_pos, right_panel_x)
        
        # ===== BOTTOM BAR =====
        bottom_bar_height = 70
        bottom_bar_y = h - bottom_bar_height
        
        cv2.rectangle(frame, (0, bottom_bar_y), (w, h), (25, 25, 25), -1)
        cv2.rectangle(frame, (0, bottom_bar_y), (w, h), (0, 200, 255), 3)
        
        # FPS (left)
        fps_color = (0, 255, 0) if self.fps > 15 else (0, 165, 255)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (25, bottom_bar_y + 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, fps_color, 2, cv2.LINE_AA)
        
        # Status (center)
        status_text = "FACE DETECTED" if face_detected else "NO FACE DETECTED"
        status_color = (0, 255, 0) if face_detected else (0, 0, 255)
        
        # Center align status
        (text_w, text_h), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        status_x = (w - text_w) // 2
        
        cv2.putText(frame, f"STATUS: {status_text}", (status_x - 60, bottom_bar_y + 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2, cv2.LINE_AA)
        
        # Mode (right)
        cv2.putText(frame, "OpenCV Mode", (w - 220, bottom_bar_y + 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2, cv2.LINE_AA)
        
        return frame
    
    def run(self):
        """Main loop"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print(" Cannot open camera!")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("\n" + "=" * 60)
        print(" EMOTION DETECTION SYSTEM - RUNNING")
        print("=" * 60)
        print("\n Camera started!")
        print("\n Controls:")
        print("   • Press 'Q' to quit")
        print("   • Press 'R' to reset blink counter")
        print("   • Press 'F' to toggle fullscreen")
        print("\n" + "=" * 60 + "\n")
        
        frame_count = 0
        fullscreen = False
        
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            
            if not ret:
                print(" Failed to capture frame")
                break
            
            # Mirror view
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(120, 120),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            face_detected = len(faces) > 0
            gestures = []
            eye_count = 0
            
            if face_detected:
                # Get largest face
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                
                # Extract face region
                face_region = gray[y:y+h, x:x+w]
                face_img = frame[y:y+h, x:x+w]
                
                # Detect gestures
                gestures, eye_count = self.detect_gestures(face_region)
                
                # Emotion detection (every 20 frames for performance)
                if frame_count % 20 == 0 and face_img.size > 0:
                    self.detect_emotion(face_img)
                
                # Draw face box
                self.draw_face_box(frame, x, y, w, h)
            
            # Draw HUD
            frame = self.draw_hud(frame, face_detected, gestures, eye_count)
            
            # Calculate FPS
            frame_time = time.time() - start_time
            self.frame_times.append(frame_time)
            if len(self.frame_times) > 0:
                avg_time = sum(self.frame_times) / len(self.frame_times)
                self.fps = 1.0 / avg_time if avg_time > 0 else 0
            
            # Display
            window_name = 'Emotion Detection System - Press Q to quit'
            cv2.imshow(window_name, frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("\n👋 Shutting down...")
                break
            elif key == ord('r') or key == ord('R'):
                self.blink_counter = 0
                print(" Blink counter reset!")
            elif key == ord('f') or key == ord('F'):
                fullscreen = not fullscreen
                if fullscreen:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n System stopped successfully!")
        print(f" Total frames processed: {frame_count}")
        print(f"  Total blinks detected: {self.blink_counter}")
        print(f" Average FPS: {self.fps:.1f}\n")

if __name__ == "__main__":
    try:
        detector = EmotionDetector()
        detector.run()
    except KeyboardInterrupt:
        print("\n  Interrupted by user")
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
