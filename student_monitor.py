import cv2
import numpy as np
import dlib
import os
from imutils import face_utils
import time
import math
import csv
from datetime import datetime

class StudentFocusMonitor:
    def __init__(self, students_dir="images", csv_log_file="dashboard/attention_log.csv", log_interval=5):
        # Initialize directories
        self.students_dir = students_dir
        self.ensure_directories_exist()
        
        # Initialize logging
        self.csv_log_file = csv_log_file
        self.log_interval = log_interval
        self.last_log_time = time.time()
        
        # Initialize dlib components
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.face_rec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
        
        # Initialize parameters
        self.distraction_threshold = 0.20
        self.consecutive_frames = 5
        self.ear_threshold = 0.22
        self.score_decay_factor = 0.95
        
        # Initialize data structures
        self.known_face_encodings = []
        self.known_face_names = []
        self.student_states = {}
        self.attention_scores = {}
        self.final_attendance_written = False
        
        # Load student data
        self.load_student_encodings()
        self.init_csv_log()
        
        # Initialize all scores to 0
        for name in self.known_face_names:
            self.attention_scores[name] = {
                "total_frames": 0,
                "focused_frames": 0,
                "score": 0.0,
                "total_focused_time": 0.0,
                "total_time": 0.0,
                "last_update": time.time()
            }

    def ensure_directories_exist(self):
        """Ensure required directories exist"""
        if not os.path.exists(self.students_dir):
            print(f"Creating directory: {self.students_dir}")
            os.makedirs(self.students_dir)
            print(f"Please add student images to '{self.students_dir}'")

    def init_csv_log(self):
        """Initialize CSV with fixed student names"""
        if not os.path.exists(self.csv_log_file):
            print(f"Initializing new CSV at {self.csv_log_file}")
            with open(self.csv_log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Name', 'Date', 'T_Attention_Time', 'T_Distraction_Time', 'Attention_Score', 'Present'])
                
                # Initialize all students with 0 values
                for name in self.known_face_names:
                    writer.writerow([
                        name,
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        0.0,  # T_Attention_Time
                        0.0,  # T_Distraction_Time
                        0.0,  # Attention_Score
                        'Absent'  # Default status
                    ])
        else:
            print(f"Using existing CSV at {self.csv_log_file}")
            # Ensure CSV matches our student list
            self._validate_csv_structure()

    def _validate_csv_structure(self):
        """Ensure CSV has all current students"""
        with open(self.csv_log_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            existing_names = {row[0] for row in reader}
        
        missing = set(self.known_face_names) - existing_names
        if missing:
            print(f"Adding missing students to CSV: {missing}")
            with open(self.csv_log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                for name in missing:
                    writer.writerow([
                        name,
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        0.0, 0.0, 0.0, 'Absent'
                    ])



# Rest of your existing methods...    
    def load_student_encodings(self):
        """Load student face encodings from images"""
        print("Loading student face encodings...")
        for filename in os.listdir(self.students_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                name = os.path.splitext(filename)[0]
                image_path = os.path.join(self.students_dir, filename)
                
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Warning: Could not load {image_path}")
                    continue
                
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                faces = self.detector(rgb_image)
                
                if len(faces) == 1:
                    shape = self.predictor(rgb_image, faces[0])
                    face_encoding = np.array(self.face_rec.compute_face_descriptor(rgb_image, shape))
                    
                    self.known_face_encodings.append(face_encoding)
                    self.known_face_names.append(name)
                    
                    # Initialize student state
                    self.student_states[name] = {
                        "status": "Initializing",
                        "focus_counter": 0,
                        "distraction_counter": 0,
                        "last_status_change": time.time()
                    }
        
        if not self.known_face_encodings:
            print("No students found. Adding test student.")
            self.known_face_names.append("Test_Student")
            self.student_states["Test_Student"] = {
                "status": "Initializing",
                "focus_counter": 0,
                "distraction_counter": 0,
                "last_status_change": time.time()
            }
    
    def get_eye_landmarks(self, shape):
        """Extract eye landmarks from the facial landmarks"""
        # Extract the indices for left and right eyes using face_utils
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        
        # Get the landmarks for the left and right eyes
        left_eye = shape[lStart:lEnd]
        right_eye = shape[rStart:rEnd]
        
        return left_eye, right_eye
    
    def calculate_eye_aspect_ratio(self, eye):
        """Calculate the eye aspect ratio to determine if the eye is open"""
        # Calculate vertical eye landmarks distances
        A = math.dist(eye[1], eye[5])
        B = math.dist(eye[2], eye[4])
        
        # Calculate horizontal eye landmarks distance
        C = math.dist(eye[0], eye[3])
        
        # Calculate eye aspect ratio
        if C == 0:  # Avoid division by zero
            return 0
            
        ear = (A + B) / (2.0 * C)
        
        return ear
    
    def analyze_eye_gaze(self, frame, shape, face_rect):
        """Analyze eye gaze direction using eye landmarks and pupil detection"""
        # Get eye landmarks
        left_eye, right_eye = self.get_eye_landmarks(shape)
        
        # Calculate eye centers
        left_eye_center = np.mean(left_eye, axis=0).astype(int)
        right_eye_center = np.mean(right_eye, axis=0).astype(int)
        
        # Calculate eye width
        left_eye_width = np.linalg.norm(left_eye[0] - left_eye[3])
        right_eye_width = np.linalg.norm(right_eye[0] - right_eye[3])
        
        # Get eye regions for pupil detection
        left_eye_region = self.extract_eye_region(frame, left_eye)
        right_eye_region = self.extract_eye_region(frame, right_eye)
        
        # Detect pupils
        left_pupil_x = self.detect_pupil(left_eye_region, int(left_eye_width))
        right_pupil_x = self.detect_pupil(right_eye_region, int(right_eye_width))
        
        # Convert to absolute coordinates
        x_min_l = np.min(left_eye[:, 0])
        x_min_r = np.min(right_eye[:, 0])
        left_iris_x = x_min_l + left_pupil_x if left_pupil_x is not None else left_eye_center[0]
        right_iris_x = x_min_r + right_pupil_x if right_pupil_x is not None else right_eye_center[0]
        
        # Calculate relative positions of irises within the eyes
        left_rel_x = (left_iris_x - left_eye_center[0]) / (left_eye_width / 2) if left_eye_width > 0 else 0
        right_rel_x = (right_iris_x - right_eye_center[0]) / (right_eye_width / 2) if right_eye_width > 0 else 0
        
        # Average relative position
        avg_rel_x = (left_rel_x + right_rel_x) / 2
        
        # Check if eyes are open enough (EAR check)
        left_ear = self.calculate_eye_aspect_ratio(left_eye)
        right_ear = self.calculate_eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2
        
        # Determine gaze direction
        if avg_ear < self.ear_threshold:  # Eyes are closed or nearly closed
            return "Eyes Closed", (left_eye_center, right_eye_center), (left_iris_x, right_iris_x)
        elif abs(avg_rel_x) < self.distraction_threshold:
            return "Focus", (left_eye_center, right_eye_center), (left_iris_x, right_iris_x)
        else:
            direction = "Left" if avg_rel_x < 0 else "Right"
            return f"Distracted ({direction})", (left_eye_center, right_eye_center), (left_iris_x, right_iris_x)
    
    def extract_eye_region(self, frame, eye):
        """Extract region of interest for the eye"""
        # Get bounding rectangle
        x_min = np.min(eye[:, 0])
        x_max = np.max(eye[:, 0])
        y_min = np.min(eye[:, 1])
        y_max = np.max(eye[:, 1])
        
        # Add margin
        x_margin = int((x_max - x_min) * 0.2)
        y_margin = int((y_max - y_min) * 0.2)
        
        # Ensure within frame
        height, width = frame.shape[:2]
        x_min = max(0, x_min - x_margin)
        y_min = max(0, y_min - y_margin)
        x_max = min(width - 1, x_max + x_margin)
        y_max = min(height - 1, y_max + y_margin)
        
        # Extract region
        eye_region = frame[y_min:y_max, x_min:x_max]
        
        return eye_region if eye_region.size > 0 else None
    
    def detect_pupil(self, eye_region, eye_width):
        """Detect pupil in eye region using image processing"""
        if eye_region is None or eye_region.size == 0:
            return None
            
        # Convert to grayscale
        gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY) if len(eye_region.shape) > 2 else eye_region
        
        # Apply blur
        gray_eye = cv2.GaussianBlur(gray_eye, (7, 7), 0)
        
        # Threshold to find dark areas (pupil)
        _, threshold = cv2.threshold(gray_eye, 40, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # Filter contours by area to find potential pupils
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 5]
        
        if not valid_contours:
            return None
            
        # Get largest contour (likely pupil)
        pupil_contour = max(valid_contours, key=cv2.contourArea)
        
        # Get center
        M = cv2.moments(pupil_contour)
        if M["m00"] == 0:
            return None
            
        cx = int(M["m10"] / M["m00"])
        
        # Return x-coordinate of pupil
        return cx
    
    def update_student_status(self, student_name, current_status):
        """Update student status based on consecutive frames"""
        if student_name not in self.student_states:
            self.student_states[student_name] = {
                "status": "Initializing",
                "focus_counter": 0,
                "distraction_counter": 0,
                "last_status_change": time.time()
            }
        
        student = self.student_states[student_name]
        
        if current_status.startswith("Focus"):
            student["focus_counter"] += 1
            student["distraction_counter"] = 0
        elif current_status.startswith("Distracted") or current_status == "Eyes Closed":
            student["distraction_counter"] += 1
            student["focus_counter"] = 0
        
        # Update status based on consecutive frames
        status_changed = False
        if student["focus_counter"] >= self.consecutive_frames and not student["status"].startswith("Focus"):
            student["status"] = "Focus"
            student["last_status_change"] = time.time()
            status_changed = True
        elif student["distraction_counter"] >= self.consecutive_frames and not student["status"].startswith("Distracted"):
            student["status"] = current_status
            student["last_status_change"] = time.time()
            status_changed = True
            
        # Update attention score
        self.update_attention_score(student_name, status_changed)
            
        return student["status"]
    
    def update_attention_score(self, student_name, status_changed):
        """Update attention score based on focus status"""
        if student_name not in self.attention_scores:
            self.attention_scores[student_name] = {
                "total_frames": 0,
                "focused_frames": 0,
                "score": 50.0,
                "total_focused_time": 0,
                "total_time": 0,
                "last_update": time.time()
            }
        
        score_data = self.attention_scores[student_name]
        status = self.student_states[student_name]["status"]
        
        # Update frame counters
        score_data["total_frames"] += 1
        if status.startswith("Focus"):
            score_data["focused_frames"] += 1
        
        # Calculate time deltas
        current_time = time.time()
        time_delta = current_time - score_data["last_update"]
        score_data["total_time"] += time_delta
        
        if status.startswith("Focus"):
            score_data["total_focused_time"] += time_delta
        
        # Update score using exponential moving average
        target_score = 100 if status.startswith("Focus") else 0
        score_data["score"] = (score_data["score"] * self.score_decay_factor + 
                               target_score * (1 - self.score_decay_factor))
        
        # Update last update time
        score_data["last_update"] = current_time
    
    def log_attention_data(self):
        current_time = time.time()
        if current_time - self.last_log_time >= self.log_interval:
            try:
                # Create temporary file
                temp_file = self.csv_log_file + '.tmp'
                
                with open(temp_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Name', 'Date', 'T_Attention_Time', 'T_Distraction_Time', 'Attention_Score', 'Present'])
                    
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    for name in self.known_face_names:
                        if name in self.attention_scores:
                            score_data = self.attention_scores[name]
                            present = 'Present' if score_data['score'] > 60 else 'Absent'
                            
                            writer.writerow([
                                name,
                                timestamp,
                                round(score_data['total_focused_time'], 2),
                                round(score_data['total_time'] - score_data['total_focused_time'], 2),
                                round(score_data['score'], 2),
                                present
                            ])
                
                # Atomic file replacement
                if os.path.exists(temp_file):
                    if os.path.exists(self.csv_log_file):
                        os.replace(temp_file, self.csv_log_file)
                    else:
                        os.rename(temp_file, self.csv_log_file)
                
                self.last_log_time = current_time
                print(f"Updated CSV at {timestamp}")
                
            except Exception as e:
                print(f"Error updating CSV: {str(e)}")
                if os.path.exists(temp_file):
                    os.remove(temp_file)
    
    def recognize_faces_and_track_focus(self, frame):
        """Recognize faces and track focus status for each student"""
        # Convert to RGB for dlib
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply histogram equalization to improve face detection in various lighting
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray_frame)
        
        # Detect faces with different parameters
        faces = self.detector(rgb_frame, 1)  # 1 means upsample once for better detection
        
        # If no faces detected with initial settings, try again with different settings
        if len(faces) == 0:
            faces = self.detector(equalized, 1)
        
        # Track faces found in this frame
        frame_faces = []
        
        for face in faces:
            # Get facial landmarks
            shape = self.predictor(rgb_frame, face)
            shape_np = face_utils.shape_to_np(shape)
            
            # Get face encoding
            face_encoding = self.face_rec.compute_face_descriptor(rgb_frame, shape)
            face_encoding = np.array(face_encoding)
            
            # Find matches
            name = "Unknown"
            if len(self.known_face_encodings) > 0:
                face_distances = np.linalg.norm(self.known_face_encodings - face_encoding, axis=1)
                best_match_index = np.argmin(face_distances)
                min_distance = face_distances[best_match_index]
                
                # Lower threshold for better recognition (adjust as needed)
                if min_distance < 0.5:  # More strict threshold
                    name = self.known_face_names[best_match_index]
            
            # If using dummy data and no match found, use test student
            if name == "Unknown" and len(self.known_face_encodings) == 0:
                name = "Test_Student"
                
            # Track that we found this student
            if name != "Unknown":
                frame_faces.append(name)
            
            # Analyze eye gaze
            gaze_status, eye_centers, iris_positions = self.analyze_eye_gaze(frame, shape_np, face)
            
            # Update student status
            if name != "Unknown":
                final_status = self.update_student_status(name, gaze_status)
            else:
                final_status = gaze_status
            
            # Draw face rectangle - green for focused, red for distracted
            status_color = (0, 255, 0) if final_status == "Focus" else (0, 0, 255)
            left, top, right, bottom = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(frame, (left, top), (right, bottom), status_color, 2)
            
            # Draw eyes
            (left_eye_center, right_eye_center) = eye_centers
            (left_iris_x, right_iris_x) = iris_positions
            
            cv2.circle(frame, tuple(left_eye_center), 3, (255, 0, 0), -1)
            cv2.circle(frame, tuple(right_eye_center), 3, (255, 0, 0), -1)
            
            # Draw irises
            cv2.circle(frame, (left_iris_x, left_eye_center[1]), 2, (0, 0, 255), -1)
            cv2.circle(frame, (right_iris_x, right_eye_center[1]), 2, (0, 0, 255), -1)
            
            # Draw name and status
            cv2.putText(frame, f"{name}", (left, top - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, f"{final_status}", (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
            
            # Draw attention score if available
            if name != "Unknown" and name in self.attention_scores:
                score = round(self.attention_scores[name]["score"], 1)
                score_color = self.get_score_color(score)
                cv2.putText(frame, f"Attention: {score}%", (left, bottom + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, score_color, 2)
                            
        # Check for students who weren't found in this frame
        for student_name in self.student_states:
            if student_name not in frame_faces:
                # If a student isn't found in frame, mark as not in frame
                self.student_states[student_name]["status"] = "Not in frame"
        
        # Draw attendance and focus stats
        self.draw_stats(frame)
        
        # Log attention data
        self.log_attention_data()
        
        return frame
    
    def get_score_color(self, score):
        """Get color based on attention score"""
        if score >= 80:
            return (0, 255, 0)  # Green
        elif score >= 50:
            return (0, 255, 255)  # Yellow
        else:
            return (0, 0, 255)  # Red
    
    def draw_stats(self, frame):
        """Draw attendance and focus statistics"""
        height, width = frame.shape[:2]
        
        # Count students
        total_students = len(self.known_face_names)
        present_students = sum(1 for state in self.student_states.values() if state["status"] != "Not in frame")
        focused_students = sum(1 for state in self.student_states.values() if state["status"] == "Focus")
        
        # Calculate class attention score
        total_score = 0
        student_count = 0
        for student_name, score_data in self.attention_scores.items():
            if self.student_states[student_name]["status"] != "Not in frame":
                total_score += score_data["score"]
                student_count += 1
        
        class_score = total_score / max(1, student_count)
        
        # Draw stats box
        cv2.rectangle(frame, (width - 250, 10), (width - 10, 110), (0, 0, 0), -1)
        cv2.rectangle(frame, (width - 250, 10), (width - 10, 110), (255, 255, 255), 1)
        
        # Draw stats text
        cv2.putText(frame, f"Total Students: {total_students}", (width - 240, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Present: {present_students}", (width - 240, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Focused: {focused_students}", (width - 240, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw class attention score
        score_color = self.get_score_color(class_score)
        cv2.putText(frame, f"Class Attention: {class_score:.1f}%", (width - 240, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, score_color, 1)

    def run(self):
        """Run the face recognition and eye tracking system"""
        print("Starting video capture...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open video capture")
            return
        
        # Set higher resolution if available
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    

        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print("Error: Can't receive frame")
                    break
                
                # Flip frame horizontally for more intuitive mirror view
                frame = cv2.flip(frame, 1)
                
                # Process frame
                processed_frame = self.recognize_faces_and_track_focus(frame)
                
                # Display result
                cv2.imshow('Student Focus Monitoring', processed_frame)
                
                # Break loop on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    
        finally:
            # Ensure resources are released and final data is written
            cap.release()
            cv2.destroyAllWindows()
            self.write_final_attendance()
            
    def write_final_attendance(self):
        """Write final attendance data"""
        if not self.final_attendance_written:
            self.log_attention_data()  # Force one last update
            self.final_attendance_written = True
            print("Final attendance data written to CSV")        

if __name__ == "__main__":
    monitor = StudentFocusMonitor(log_interval=5)  # Log every 5 seconds
    monitor.run()