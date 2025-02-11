import cv2
import numpy as np
import json
from typing import Dict, Any
import os

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

    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """Analyze image using OpenCV and return metadata."""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return {}
            
            # Resize large images to improve performance
            max_dimension = 1024
            height, width = img.shape[:2]
            if max(height, width) > max_dimension:
                scale = max_dimension / max(height, width)
                img = cv2.resize(img, None, fx=scale, fy=scale)
                height, width = img.shape[:2]
            
            analysis = {
                'height': int(height),
                'width': int(width)
            }
            
            # Convert to grayscale once and cache
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces using DNN
            faces = self._detect_faces_dnn(img)
            analysis['faces_detected'] = len(faces)
            
            # Process color analysis in parallel with face detection
            color_analysis = self._extract_colors_optimized(img)
            analysis.update(color_analysis)
            
            # Calculate image quality metrics
            quality_metrics = self._calculate_quality_metrics(gray)
            analysis.update(quality_metrics)
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing image with OpenCV: {e}")
            return {}

    def _detect_faces_dnn(self, img: np.ndarray, confidence_threshold: float = 0.5) -> list:
        """Detect faces using DNN model."""
        blob = cv2.dnn.blobFromImage(
            img, 1.0, (300, 300), [104, 117, 123], False, False
        )
        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confidence_threshold:
                faces.append(detections[0, 0, i])
        
        return faces

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