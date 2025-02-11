import cv2
import numpy as np
import json
from typing import Dict, Any, List
import os
from functools import lru_cache
import concurrent.futures
from pathlib import Path

class OpenCVService:
    def __init__(self):
        # Use DNN face detector instead of cascade classifier
        models_dir = os.path.expanduser("~/.opencv_models")
        os.makedirs(models_dir, exist_ok=True)
        
        self.model_file = os.path.join(models_dir, "res10_300x300_ssd_iter_140000.caffemodel")
        self.config_file = os.path.join(models_dir, "deploy.prototxt")
        
        # Download models if they don't exist
        self._ensure_models_exist()
        
        # Load DNN model
        self.face_detector = cv2.dnn.readNet(self.model_file, self.config_file)
        
        # Use OpenCL if available
        if cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(True)
        
        # Optimize for CPU
        self.face_detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.face_detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self._cache = {}  # Simple cache for results

        # Load the face detection classifier
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def _ensure_models_exist(self):
        """Download required models if they don't exist."""
        models_dir = os.path.expanduser("~/.opencv_models")
        os.makedirs(models_dir, exist_ok=True)
        
        # Model files URLs
        model_urls = {
            "res10_300x300_ssd_iter_140000.caffemodel": 
                "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
            "deploy.prototxt":
                "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
        }
        
        # Download missing models
        for filename, url in model_urls.items():
            filepath = os.path.join(models_dir, filename)
            if not os.path.exists(filepath):
                import urllib.request
                print(f"Downloading {filename}...")
                urllib.request.urlretrieve(url, filepath)

    @lru_cache(maxsize=100)
    def _detect_faces_dnn(self, image_hash: str) -> list:
        """Cache face detection results using image hash."""
        img = self._cache.get(image_hash)
        if img is None:
            return []
            
        blob = cv2.dnn.blobFromImage(
            img, 1.0, (300, 300), [104, 117, 123], False, False
        )
        self.face_detector.setInput(blob)
        return self.face_detector.forward()

    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """Analyze image using OpenCV and return metadata."""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return {}
                
            # Create hash of image for caching
            image_hash = hash(img.tobytes())
            self._cache[image_hash] = img
            
            # Parallel processing for different analyses
            with concurrent.futures.ThreadPoolExecutor() as executor:
                face_future = executor.submit(self._detect_faces_dnn, image_hash)
                color_future = executor.submit(self._extract_colors_optimized, img)
                quality_future = executor.submit(self._calculate_quality_metrics, 
                    cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
                
            analysis = {
                'height': int(img.shape[0]),
                'width': int(img.shape[1]),
                'faces_detected': len(face_future.result()),
                **color_future.result(),
                **quality_future.result()
            }
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing image with OpenCV: {e}")
            return {}
        finally:
            # Clean up cache
            if image_hash in self._cache:
                del self._cache[image_hash]

    def _extract_colors_optimized(self, img: np.ndarray) -> Dict[str, Any]:
        """Extract color information using optimized method."""
        # Downsample image for color analysis
        height, width = img.shape[:2]
        pixels = img[::4, ::4].reshape(-1, 3)  # Sample every 4th pixel
        pixels = np.float32(pixels)
        
        # Use more efficient k-means parameters
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        flags = cv2.KMEANS_PP_CENTERS  # Use k-means++ initialization
        
        # Perform k-means clustering
        _, labels, palette = cv2.kmeans(pixels, 5, None, criteria, 1, flags)
        
        # Convert colors to hex format
        dominant_colors = [
            '#{:02x}{:02x}{:02x}'.format(int(c[2]), int(c[1]), int(c[0]))
            for c in palette
        ]
        
        # Calculate average color more efficiently
        avg_color = img.mean(axis=(0,1))
        avg_hex = '#{:02x}{:02x}{:02x}'.format(
            int(avg_color[2]), int(avg_color[1]), int(avg_color[0])
        )
        
        return {
            'dominant_colors': json.dumps(dominant_colors),
            'average_color': avg_hex
        }

    def _calculate_quality_metrics(self, gray: np.ndarray) -> Dict[str, Any]:
        """Calculate image quality metrics efficiently."""
        # Use smaller kernel for Laplacian
        laplacian_var = cv2.Laplacian(gray, cv2.CV_32F, ksize=1).var()
        
        # Calculate brightness more efficiently
        brightness = gray.mean()
        
        return {
            'is_blurry': bool(laplacian_var < 100),
            'brightness_score': float(brightness)
        }

    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Optimize image for processing."""
        # Convert to float32 for better precision
        img = img.astype(np.float32)
        
        # Normalize
        img /= 255.0
        
        # Apply CLAHE for better contrast
        if len(img.shape) > 2:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            lab = cv2.merge((l,a,b))
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return img 

    def _optimize_memory(self, img: np.ndarray, max_size: int = 1024) -> np.ndarray:
        """Optimize memory usage for large images."""
        h, w = img.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            return cv2.resize(img, None, fx=scale, fy=scale, 
                             interpolation=cv2.INTER_AREA)
        return img 

    def analyze_video(self, video_path: str) -> Dict[Any, Any]:
        """Analyze video file and extract various metrics."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file")

            # Get basic video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Initialize metrics
            face_counts = []
            motion_scores = []
            brightness_values = []
            scene_changes = []
            prev_frame = None
            frame_count = 0
            sample_interval = max(1, total_frames // 100)  # Sample every 1% of frames

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Only analyze every nth frame for performance
                if frame_count % sample_interval == 0:
                    # Detect faces
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                    face_counts.append(len(faces))

                    # Calculate brightness
                    brightness = np.mean(gray)
                    brightness_values.append(brightness)

                    # Detect motion if we have a previous frame
                    if prev_frame is not None:
                        motion_score = self._calculate_motion(prev_frame, gray)
                        motion_scores.append(motion_score)

                        # Detect scene changes
                        if motion_score > 50:  # Threshold for scene change
                            scene_changes.append(frame_count / fps)  # Store timestamp

                    prev_frame = gray

                frame_count += 1

            cap.release()

            # Calculate average metrics
            avg_faces = np.mean(face_counts) if face_counts else 0
            avg_motion = np.mean(motion_scores) if motion_scores else 0
            avg_brightness = np.mean(brightness_values) if brightness_values else 0
            is_stable = avg_motion < 30  # Threshold for stability

            # Analyze shot types based on motion and faces
            shot_types = self._analyze_shot_types(motion_scores, face_counts)

            # Camera motion analysis
            camera_motion = self._analyze_camera_motion(motion_scores)

            return {
                "fps": float(fps),
                "total_frames": int(total_frames),
                "width": int(width),
                "height": int(height),
                "faces_detected": int(avg_faces),
                "motion_score": float(avg_motion),
                "brightness_score": float(avg_brightness),
                "is_stable": bool(avg_motion < 30),  # Convert to Python bool
                "scene_changes": [float(ts) for ts in scene_changes],  # Convert to list of floats
                "shot_types": {
                    "close_up_percentage": float(shot_types["close_up_percentage"]),
                    "wide_shot_percentage": float(shot_types["wide_shot_percentage"]),
                    "action_shot_percentage": float(shot_types["action_shot_percentage"]),
                    "static_shot_percentage": float(shot_types["static_shot_percentage"])
                },
                "camera_motion": {
                    "static_percentage": float(camera_motion["static_percentage"]),
                    "pan_tilt_percentage": float(camera_motion["pan_tilt_percentage"]),
                    "tracking_percentage": float(camera_motion["tracking_percentage"]),
                    "rapid_movement_percentage": float(camera_motion["rapid_movement_percentage"])
                }
            }

        except Exception as e:
            print(f"Error analyzing video: {str(e)}")
            return {}

    def _calculate_motion(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> float:
        """Calculate motion between two frames using absolute difference."""
        if prev_frame is None:
            return 0.0

        # Convert current frame to grayscale if it isn't already
        if len(curr_frame.shape) == 3:
            curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # Calculate absolute difference and mean
        diff = cv2.absdiff(prev_frame, curr_frame)
        return np.mean(diff)

    def _analyze_shot_types(self, motion_scores: List[float], face_counts: List[int]) -> Dict[str, float]:
        """Analyze shot types based on motion and face detection."""
        if not motion_scores or not face_counts:
            return {}

        # Convert to numpy arrays for easier analysis
        motion = np.array(motion_scores)
        faces = np.array(face_counts)

        # Define shot type criteria
        close_up = np.mean(faces > 0) * 100  # Percentage of frames with faces
        wide_shot = np.mean(faces == 0) * 100  # Percentage of frames without faces
        action_shot = np.mean(motion > 40) * 100  # Percentage of high motion frames
        static_shot = np.mean(motion < 10) * 100  # Percentage of low motion frames

        return {
            "close_up_percentage": float(close_up),
            "wide_shot_percentage": float(wide_shot),
            "action_shot_percentage": float(action_shot),
            "static_shot_percentage": float(static_shot)
        }

    def _analyze_camera_motion(self, motion_scores: List[float]) -> Dict[str, float]:
        """Analyze camera motion patterns."""
        if not motion_scores:
            return {}

        motion = np.array(motion_scores)
        
        # Define motion thresholds
        static = np.mean(motion < 5) * 100
        pan_tilt = np.mean((motion >= 5) & (motion < 20)) * 100
        tracking = np.mean((motion >= 20) & (motion < 40)) * 100
        rapid = np.mean(motion >= 40) * 100

        return {
            "static_percentage": float(static),
            "pan_tilt_percentage": float(pan_tilt),
            "tracking_percentage": float(tracking),
            "rapid_movement_percentage": float(rapid)
        }