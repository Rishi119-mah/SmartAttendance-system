import cv2
import os
import json
import numpy as np
import threading

class FaceRecognizer:
    def __init__(self):
        self.lock = threading.Lock()
        
        # Load Haar Cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_detector = cv2.CascadeClassifier(cascade_path)
        
        # Init LBPH Face Recognizer
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        self.dataset_dir = "dataset"
        self.models_dir = "models"
        self.trainer_path = os.path.join(self.models_dir, "trainer.yml")
        self.label_map_path = os.path.join(self.models_dir, "label_map.json")
        
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
            
        self.label_map = {}
        # Load existing model if available
        if os.path.exists(self.trainer_path) and os.path.exists(self.label_map_path):
            try:
                self.recognizer.read(self.trainer_path)
                with open(self.label_map_path, 'r') as f:
                    self.label_map = json.load(f)
            except Exception as e:
                print(f"Error loading model: {e}")

    def process_registration_frame(self, name, count, frame):
        """
        Takes a single frame from the frontend, extracts the face, and saves it.
        """
        with self.lock:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
            
            # Process only if at least one face is found
            for (x, y, w, h) in faces:
                face_img = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face_img, (200, 200))
                
                # Save to dataset
                file_path = os.path.join(self.dataset_dir, f"{name}_{count}.jpg")
                cv2.imwrite(file_path, face_resized)
                
                # Return True after saving the first valid face in the frame
                return True
                
            return False

    def train_model(self):
        """
        Read all images from dataset/, parse name from filename,
        build label map {name: int_id}, train LBPH on all images,
        save model to models/trainer.yml, save label map as JSON
        """
        with self.lock:
            image_paths = [os.path.join(self.dataset_dir, f) for f in os.listdir(self.dataset_dir) if f.endswith('.jpg')]
            
            if not image_paths:
                return False
                
            face_samples = []
            ids = []
            current_id = 0
            name_to_id = {}
            
            for image_path in image_paths:
                try:
                    gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if gray_img is None:
                        continue
                        
                    filename = os.path.split(image_path)[-1]
                    name = filename.split('_')[0]
                    
                    if name not in name_to_id:
                        name_to_id[name] = current_id
                        current_id += 1
                        
                    face_samples.append(gray_img)
                    ids.append(name_to_id[name])
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    
            if not face_samples:
                return False
                
            self.recognizer.train(face_samples, np.array(ids, dtype=np.int32))
            self.recognizer.write(self.trainer_path)
            
            # Save label map
            id_to_name = {v: k for k, v in name_to_id.items()}
            with open(self.label_map_path, 'w') as f:
                json.dump(id_to_name, f)
                
            self.label_map = id_to_name
            return True

    def recognize_frame(self, frame):
        """
        Detect faces in grayscale frame, for each face run LBPH predict,
        confidence threshold 70 → known/unknown,
        return list of {name, confidence, x, y, w, h}
        """
        with self.lock:
            results = []
            if not self.label_map:
                return results # Model not trained or loaded
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
            
            for (x, y, w, h) in faces:
                face_img = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face_img, (200, 200))
                
                try:
                    id_, distance = self.recognizer.predict(face_resized)
                    
                    print(f"DEBUG: Predicted ID {id_} with distance {distance}")
                    
                    # LBPH distance: 0 is perfect, > 75 is usually unknown or risky
                    if distance < 75:
                        name = self.label_map.get(str(id_)) or self.label_map.get(id_, "Unknown")
                        # Map 0 distance to 100% conf, 75 distance to ~31% conf
                        confidence = int(max(0, 100 - (distance * 100 / 110)))
                    else:
                        name = "Unknown"
                        confidence = 0
                        
                    results.append({
                        "name": name,
                        "confidence": confidence,
                        "x": int(x),
                        "y": int(y),
                        "w": int(w),
                        "h": int(h)
                    })
                except Exception as e:
                    print(f"Prediction error: {e}")
                    
            return results
