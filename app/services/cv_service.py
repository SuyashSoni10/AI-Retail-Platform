import os
import io
import pickle
import random
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.layers import Layer

# Custom layers to support loading model serialized on newer Keras/TensorFlow versions
class TrueDivide(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, x, y=1.0):
        return x / y
    def __call__(self, *args, **kwargs):
        tensor = args[0]
        divisor = args[1] if len(args) > 1 else 1.0
        kwargs['y'] = divisor
        return super().__call__(tensor, **kwargs)

class CustomSubtract(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, x, y=0.0):
        return x - y
    def __call__(self, *args, **kwargs):
        tensor = args[0]
        subtrahend = args[1] if len(args) > 1 else 0.0
        kwargs['y'] = subtrahend
        return super().__call__(tensor, **kwargs)

class CVService:
    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, 'models', 'product_classifier.h5')
        face_db_path = os.path.join(base_dir, 'models', 'face_db.pkl')

        print(f"Loading Product Classifier model from {model_path}...")
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                'TrueDivide': TrueDivide,
                'Subtract': CustomSubtract
            }
        )
        self.classes = ['bags', 'clothing', 'electronics', 'groceries', 'shoes']

        print(f"Loading Face Database from {face_db_path}...")
        if os.path.exists(face_db_path):
            with open(face_db_path, 'rb') as f:
                self.face_db = pickle.load(f)
        else:
            self.face_db = {"label_map": {0: "Alice", 1: "Bob", 2: "Charlie"}, "engine": "simulated"}

    def classify_product(self, image_bytes: bytes) -> dict:
        try:
            # Decode and resize image
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            img = img.resize((128, 128))
            
            # Convert to numpy array and preprocess (preprocess_input scale: -1 to 1)
            img_array = np.array(img, dtype=np.float32)
            img_array = (img_array / 127.5) - 1.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            preds = self.model.predict(img_array)
            class_idx = np.argmax(preds[0])
            confidence = float(preds[0][class_idx])
            category = self.classes[class_idx]

            return {
                "category": category,
                "confidence": confidence
            }
        except Exception as e:
            print("Error in classify_product:", e)
            return {"category": "unknown", "confidence": 0.0, "error": str(e)}

    def recognize_face(self, image_bytes: bytes) -> dict:
        try:
            engine = self.face_db.get('engine', 'simulated')
            
            if engine == 'simulated':
                # Simulated fallback face recognition
                label_map = self.face_db.get('label_map', {0: "Alice", 1: "Bob", 2: "Charlie"})
                names = list(label_map.values())
                matched_name = random.choice(names)
                
                # Mock visit logging
                visit_logged = True
                
                return {
                    "name": matched_name,
                    "matched": True,
                    "logged": visit_logged,
                    "engine": "simulated"
                }
            else:
                # Real face recognition engines can be expanded here if libraries are loaded
                return {
                    "name": "Unknown",
                    "matched": False,
                    "logged": False,
                    "engine": engine
                }
        except Exception as e:
            print("Error in recognize_face:", e)
            return {"name": "Unknown", "matched": False, "logged": False, "error": str(e)}

# Instantiate singleton service instance
cv_service = CVService()
