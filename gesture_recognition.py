import cv2
import numpy as np
import mediapipe as mp
import logging
import json
import os

logger = logging.getLogger(__name__)

class GestureRecognizer:
    """
    A class that handles hand gesture recognition using MediaPipe.
    """
    
    def __init__(self):
        """Initialize the gesture recognizer with MediaPipe Hands and Face."""
        self.mp_hands = mp.solutions.hands
        self.mp_face = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        self.face_detection = self.mp_face.FaceDetection(
            min_detection_confidence=0.5,
            model_selection=0
        )
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Path to custom gestures JSON file
        self.custom_gestures_file = 'custom_gestures.json'
        self.custom_gestures = self._load_custom_gestures()
        
        # Define gestures and their corresponding landmark patterns
        self.gestures = {
            # Basic gestures
            "thumbs_up": self._is_thumbs_up,
            "thumbs_down": self._is_thumbs_down,
            "palm_open": self._is_palm_open,
            "fist": self._is_fist,
            "peace": self._is_peace_sign,
            "pointer": self._is_pointer_finger,
            "ok_sign": self._is_ok_sign,
            "rock_on": self._is_rock_on,
            "pinch": self._is_pinch,
            "swipe_left": self._is_swipe_left,
            # Additional gestures
            "victory": self._is_victory_sign,
            "call_me": self._is_call_me,
            "heart_hand": self._is_heart_hand,
            "high_five": self._is_high_five,
            "wave_hello": self._is_wave_hello,
            "fingers_crossed": self._is_fingers_crossed,
            "horns": self._is_horns,
            "stop": self._is_stop_sign,
            "clap": self._is_clap,
            "handshake": self._is_handshake
        }
        
        logger.debug("GestureRecognizer initialized with MediaPipe Hands")
    
    def _load_custom_gestures(self):
        """Load custom gestures from JSON file if it exists."""
        if os.path.exists(self.custom_gestures_file):
            try:
                with open(self.custom_gestures_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading custom gestures: {e}")
                return []
        else:
            # Create empty file if it doesn't exist
            with open(self.custom_gestures_file, 'w') as f:
                json.dump([], f)
            return []
    
    def save_custom_gesture(self, name, description, sample_landmarks=None):
        """Save a new custom gesture to the JSON file."""
        if not name or not description:
            return False, "Name and description are required"
            
        # Clean up the name (lowercase with underscores)
        sanitized_name = name.lower().replace(' ', '_')
        
        # Check if this name already exists in built-in gestures
        built_in_names = [g["name"] for g in self.get_built_in_gestures()]
        if sanitized_name in built_in_names:
            return False, f"A built-in gesture with name '{sanitized_name}' already exists"
        
        # Check if this name already exists in custom gestures
        custom_names = [g["name"] for g in self.custom_gestures]
        if sanitized_name in custom_names:
            return False, f"A custom gesture with name '{sanitized_name}' already exists"
        
        # Add the new gesture
        new_gesture = {
            "name": sanitized_name,
            "description": description,
            "is_custom": True
        }
        
        # Add sample landmarks if provided
        if sample_landmarks is not None:
            new_gesture["sample_landmarks"] = sample_landmarks
        
        self.custom_gestures.append(new_gesture)
        
        # Save to file
        try:
            with open(self.custom_gestures_file, 'w') as f:
                json.dump(self.custom_gestures, f, indent=2)
            return True, f"Custom gesture '{sanitized_name}' added successfully"
        except Exception as e:
            logger.error(f"Error saving custom gesture: {e}")
            return False, f"Error saving custom gesture: {str(e)}"
    
    def delete_custom_gesture(self, name):
        """Delete a custom gesture by name."""
        original_count = len(self.custom_gestures)
        self.custom_gestures = [g for g in self.custom_gestures if g["name"] != name]
        
        if len(self.custom_gestures) < original_count:
            # Save updated list to file
            try:
                with open(self.custom_gestures_file, 'w') as f:
                    json.dump(self.custom_gestures, f, indent=2)
                return True, f"Custom gesture '{name}' deleted successfully"
            except Exception as e:
                logger.error(f"Error saving custom gestures after deletion: {e}")
                return False, f"Error saving changes: {str(e)}"
        else:
            return False, f"No custom gesture found with name '{name}'"
    
    def get_built_in_gestures(self):
        """Return a list of built-in supported gestures with descriptions."""
        return [
            {"name": "thumbs_up", "description": "Thumb pointing upward, other fingers closed", "is_custom": False},
            {"name": "thumbs_down", "description": "Thumb pointing downward, other fingers closed", "is_custom": False},
            {"name": "palm_open", "description": "Open palm with fingers spread", "is_custom": False},
            {"name": "fist", "description": "Closed hand forming a fist", "is_custom": False},
            {"name": "peace", "description": "Index and middle fingers forming a V, other fingers closed", "is_custom": False},
            {"name": "pointer", "description": "Index finger extended, other fingers closed", "is_custom": False},
            {"name": "ok_sign", "description": "Thumb and index finger forming a circle, other fingers extended", "is_custom": False},
            {"name": "rock_on", "description": "Index and pinky fingers extended, other fingers closed", "is_custom": False},
            {"name": "pinch", "description": "Thumb and index finger close together, other fingers extended", "is_custom": False},
            {"name": "swipe_left", "description": "Hand moving from right to left with palm facing left", "is_custom": False}
        ]
    
    def get_supported_gestures(self):
        """Return a list of all supported gestures (built-in + custom)."""
        all_gestures = self.get_built_in_gestures() + self.custom_gestures
        return all_gestures
    
    def process_frame(self, frame):
        """
        Process a single frame to detect hands, faces and recognize gestures.
        
        Args:
            frame: A BGR image frame from the camera
            
        Returns:
            Tuple of (processed_frame, detected_gestures, face_status)
        """
        if frame is None:
            logger.error("Received empty frame for processing")
            return None, [], "unknown"
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe Hands
        hand_results = self.hands.process(rgb_frame)
        
        # Process face detection
        face_results = self.face_mesh.process(rgb_frame)
        
        # Convert back to BGR for display
        processed_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        # Determine face activity status
        face_status = self._analyze_face_activity(face_results, processed_frame)
        
        detected_gestures = []
        
        # Check if hands are detected
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Draw hand landmarks on the frame
                self.mp_drawing.draw_landmarks(
                    processed_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Convert landmarks to numpy array for easier processing
                landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                
                # Check for each gesture
                for gesture_name, gesture_detector in self.gestures.items():
                    if gesture_detector(landmarks_array):
                        detected_gestures.append(gesture_name)
                        # Draw gesture name on frame
                        wrist_pos = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                        h, w, _ = processed_frame.shape
                        x, y = int(wrist_pos.x * w), int(wrist_pos.y * h)
                        cv2.putText(processed_frame, gesture_name, (x, y - 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        break  # Only detect one gesture per hand
        
        return processed_frame, detected_gestures
    
    def _is_thumbs_up(self, landmarks):
        """Check if the hand gesture is thumbs up."""
        thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = landmarks[self.mp_hands.HandLandmark.THUMB_IP]
        index_mcp = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        
        # Thumb should be pointing upward and other fingers should be closed
        if (thumb_tip[1] < thumb_ip[1] and  # Y-coordinate comparison for upward direction
            thumb_tip[0] > index_mcp[0]):   # X-coordinate comparison for thumb position
            return True
        return False
    
    def _is_thumbs_down(self, landmarks):
        """Check if the hand gesture is thumbs down."""
        thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = landmarks[self.mp_hands.HandLandmark.THUMB_IP]
        index_mcp = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        
        # Thumb should be pointing downward and other fingers should be closed
        if (thumb_tip[1] > thumb_ip[1] and  # Y-coordinate comparison for downward direction
            thumb_tip[0] > index_mcp[0]):   # X-coordinate comparison for thumb position
            return True
        return False
    
    def _is_palm_open(self, landmarks):
        """Check if the hand gesture is open palm with fingers spread."""
        # Calculate finger straightness by measuring distance between tip and MCP
        index_straight = self._is_finger_straight(landmarks, 
                                                 self.mp_hands.HandLandmark.INDEX_FINGER_TIP, 
                                                 self.mp_hands.HandLandmark.INDEX_FINGER_MCP)
        middle_straight = self._is_finger_straight(landmarks, 
                                                 self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP, 
                                                 self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP)
        ring_straight = self._is_finger_straight(landmarks, 
                                               self.mp_hands.HandLandmark.RING_FINGER_TIP, 
                                               self.mp_hands.HandLandmark.RING_FINGER_MCP)
        pinky_straight = self._is_finger_straight(landmarks, 
                                                self.mp_hands.HandLandmark.PINKY_TIP, 
                                                self.mp_hands.HandLandmark.PINKY_MCP)
        
        # All fingers should be extended
        if index_straight and middle_straight and ring_straight and pinky_straight:
            return True
        return False
    
    def _is_fist(self, landmarks):
        """Check if the hand gesture is a closed fist."""
        # Check if all finger tips are below their respective MCP joints
        index_closed = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP][1] > landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_MCP][1]
        middle_closed = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP][1] > landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP][1]
        ring_closed = landmarks[self.mp_hands.HandLandmark.RING_FINGER_TIP][1] > landmarks[self.mp_hands.HandLandmark.RING_FINGER_MCP][1]
        pinky_closed = landmarks[self.mp_hands.HandLandmark.PINKY_TIP][1] > landmarks[self.mp_hands.HandLandmark.PINKY_MCP][1]
        
        if index_closed and middle_closed and ring_closed and pinky_closed:
            return True
        return False
    
    def _is_peace_sign(self, landmarks):
        """Check if the hand gesture is a peace sign (V shape with index and middle fingers)."""
        index_extended = self._is_finger_extended(landmarks, 
                                               self.mp_hands.HandLandmark.INDEX_FINGER_TIP, 
                                               self.mp_hands.HandLandmark.INDEX_FINGER_MCP)
        middle_extended = self._is_finger_extended(landmarks, 
                                                self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP, 
                                                self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP)
        ring_closed = not self._is_finger_extended(landmarks, 
                                               self.mp_hands.HandLandmark.RING_FINGER_TIP, 
                                               self.mp_hands.HandLandmark.RING_FINGER_MCP)
        pinky_closed = not self._is_finger_extended(landmarks, 
                                                self.mp_hands.HandLandmark.PINKY_TIP, 
                                                self.mp_hands.HandLandmark.PINKY_MCP)
        
        if index_extended and middle_extended and ring_closed and pinky_closed:
            return True
        return False
    
    def _is_pointer_finger(self, landmarks):
        """Check if only the index finger is extended (pointing gesture)."""
        index_extended = self._is_finger_extended(landmarks, 
                                               self.mp_hands.HandLandmark.INDEX_FINGER_TIP, 
                                               self.mp_hands.HandLandmark.INDEX_FINGER_MCP)
        middle_closed = not self._is_finger_extended(landmarks, 
                                                 self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP, 
                                                 self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP)
        ring_closed = not self._is_finger_extended(landmarks, 
                                               self.mp_hands.HandLandmark.RING_FINGER_TIP, 
                                               self.mp_hands.HandLandmark.RING_FINGER_MCP)
        pinky_closed = not self._is_finger_extended(landmarks, 
                                                self.mp_hands.HandLandmark.PINKY_TIP, 
                                                self.mp_hands.HandLandmark.PINKY_MCP)
        
        if index_extended and middle_closed and ring_closed and pinky_closed:
            return True
        return False
    
    def _is_ok_sign(self, landmarks):
        """Check if the hand is making an OK sign (thumb and index finger forming a circle)."""
        thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        
        # Calculate distance between thumb tip and index tip
        distance = np.linalg.norm(thumb_tip - index_tip)
        
        # If the distance is small, thumb and index are likely forming a circle
        if distance < 0.08:  # Threshold may need to be adjusted
            return True
        return False
    
    def _is_rock_on(self, landmarks):
        """Check if the hand is making a rock on sign (index and pinky extended)."""
        index_extended = self._is_finger_extended(landmarks, 
                                               self.mp_hands.HandLandmark.INDEX_FINGER_TIP, 
                                               self.mp_hands.HandLandmark.INDEX_FINGER_MCP)
        middle_closed = not self._is_finger_extended(landmarks, 
                                                 self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP, 
                                                 self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP)
        ring_closed = not self._is_finger_extended(landmarks, 
                                               self.mp_hands.HandLandmark.RING_FINGER_TIP, 
                                               self.mp_hands.HandLandmark.RING_FINGER_MCP)
        pinky_extended = self._is_finger_extended(landmarks, 
                                               self.mp_hands.HandLandmark.PINKY_TIP, 
                                               self.mp_hands.HandLandmark.PINKY_MCP)
        
        if index_extended and middle_closed and ring_closed and pinky_extended:
            return True
        return False
    
    def _is_pinch(self, landmarks):
        """Check if the hand is making a pinch gesture (thumb and index close together)."""
        thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        
        # Calculate distance between thumb tip and index tip
        distance = np.linalg.norm(thumb_tip - index_tip)
        
        # Check if other fingers are extended
        middle_extended = self._is_finger_extended(landmarks, 
                                                self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP, 
                                                self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP)
        ring_extended = self._is_finger_extended(landmarks, 
                                              self.mp_hands.HandLandmark.RING_FINGER_TIP, 
                                              self.mp_hands.HandLandmark.RING_FINGER_MCP)
        pinky_extended = self._is_finger_extended(landmarks, 
                                               self.mp_hands.HandLandmark.PINKY_TIP, 
                                               self.mp_hands.HandLandmark.PINKY_MCP)
        
        # If the distance is small and other fingers are extended
        if distance < 0.08 and middle_extended and ring_extended and pinky_extended:  
            return True
        return False
    
    def _is_victory_sign(self, landmarks):
        """Check if hand is making victory sign."""
        return self._is_peace_sign(landmarks)
        
    def _is_call_me(self, landmarks):
        """Check if hand is making call me gesture."""
        thumb_extended = self._is_finger_extended(landmarks, 
                                              self.mp_hands.HandLandmark.THUMB_TIP,
                                              self.mp_hands.HandLandmark.THUMB_MCP)
        pinky_extended = self._is_finger_extended(landmarks,
                                              self.mp_hands.HandLandmark.PINKY_TIP,
                                              self.mp_hands.HandLandmark.PINKY_MCP)
        return thumb_extended and pinky_extended
        
    def _is_heart_hand(self, landmarks):
        """Check if hands are making heart shape."""
        thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        distance = np.linalg.norm(thumb_tip - index_tip)
        return distance < 0.1
        
    def _is_high_five(self, landmarks):
        """Check if hand is making high five gesture."""
        return self._is_palm_open(landmarks)
        
    def _is_wave_hello(self, landmarks):
        """Check if hand is waving hello."""
        return self._is_palm_open(landmarks)
        
    def _is_fingers_crossed(self, landmarks):
        """Check if fingers are crossed."""
        index_tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        distance = np.linalg.norm(index_tip - middle_tip)
        return distance < 0.05
        
    def _is_horns(self, landmarks):
        """Check if hand is making horns gesture."""
        index_extended = self._is_finger_extended(landmarks,
                                              self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
                                              self.mp_hands.HandLandmark.INDEX_FINGER_MCP)
        pinky_extended = self._is_finger_extended(landmarks,
                                              self.mp_hands.HandLandmark.PINKY_TIP,
                                              self.mp_hands.HandLandmark.PINKY_MCP)
        return index_extended and pinky_extended
        
    def _is_stop_sign(self, landmarks):
        """Check if hand is making stop sign."""
        return self._is_palm_open(landmarks)
        
    def _is_clap(self, landmarks):
        """Check if hands are clapping."""
        return self._is_palm_open(landmarks)
        
    def _is_handshake(self, landmarks):
        """Check if hand is in handshake position."""
        return self._is_palm_open(landmarks)

    def _is_swipe_left(self, landmarks):
        """Check if the hand is in a position that could be a swipe left gesture."""
        # This is a simplistic implementation that detects a hand position 
        # that would be consistent with a swipe left motion
        
        # Check if palm is facing left
        wrist = landmarks[self.mp_hands.HandLandmark.WRIST]
        middle_mcp = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        
        # Palm facing left means middle_mcp is to the left of wrist
        if middle_mcp[0] < wrist[0]:
            # Check if fingers are relatively straight
            if self._is_palm_open(landmarks):
                return True
        return False
    
    def _is_finger_extended(self, landmarks, tip_idx, mcp_idx):
        """Check if a finger is extended by comparing the y-coordinates of the tip and MCP."""
        return landmarks[tip_idx][1] < landmarks[mcp_idx][1]
    
    def _is_finger_straight(self, landmarks, tip_idx, mcp_idx):
        """Check if a finger is straight by calculating the distance between tip and MCP in 3D space."""
        tip = landmarks[tip_idx]
        mcp = landmarks[mcp_idx]
        
        # Calculate Euclidean distance
        distance = np.linalg.norm(tip - mcp)
        
        # If the distance is greater than a threshold, the finger is likely straight
        # This threshold may need adjustment based on hand size variance
        if distance > 0.1:  
            return True
        return False


    def _analyze_face_activity(self, face_results, frame):
        """Analyze face activity to detect if user is active, sleepy, or looking away."""
        if not face_results.multi_face_landmarks:
            cv2.putText(frame, "No Face Detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return "no_face"
            
        face_landmarks = face_results.multi_face_landmarks[0]
        
        # Get eye landmarks for better detection
        left_eye = face_landmarks.landmark[33]  # Left eye
        right_eye = face_landmarks.landmark[133]  # Right eye
        nose = face_landmarks.landmark[1]  # Nose tip
        
        # Calculate eye aspect ratio and head pose
        eye_ratio = abs(left_eye.y - right_eye.y)
        head_pose = abs(nose.x - 0.5)  # Check if face is centered
        
        if head_pose > 0.3:
            cv2.putText(frame, "Looking Away", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            return "looking_away"
        elif eye_ratio < 0.02:
            cv2.putText(frame, "Sleepy", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return "sleepy"
        else:
            cv2.putText(frame, "Active", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return "active"
            
        face_landmarks = face_results.multi_face_landmarks[0]
        
        # Get eye landmarks
        left_eye = face_landmarks.landmark[159]  # Left eye bottom
        right_eye = face_landmarks.landmark[386]  # Right eye bottom
        
        # Calculate eye aspect ratio
        eye_ratio = abs(left_eye.y - right_eye.y)
        
        # Draw face mesh
        self.mp_drawing.draw_landmarks(
            frame,
            face_landmarks,
            self.mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
        )
        
        # Determine status based on eye ratio
        if eye_ratio < 0.02:  # Eyes nearly closed
            cv2.putText(frame, "Sleepy", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return "sleepy"
        else:
            cv2.putText(frame, "Active", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return "active"
