import os
import json
from PIL import Image
import io

class FaceRecognitionService:
    def __init__(self):
        pass
    
    def extract_embedding(self, image_path):
        try:
            image = Image.open(image_path).convert('L')
            image = image.resize((100, 100))
            pixels = list(image.getdata())
            embedding = [p / 255.0 for p in pixels]
            return json.dumps(embedding)
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return None
    
    def compare_faces(self, embedding1, embedding2):
        try:
            emb1 = json.loads(embedding1)
            emb2 = json.loads(embedding2)
            
            if len(emb1) != len(emb2):
                return 0.0
            
            diff_sum = sum((a - b) ** 2 for a, b in zip(emb1, emb2))
            similarity = max(0, 1 - (diff_sum / len(emb1)))
            return similarity
        except Exception as e:
            print(f"Error comparing faces: {e}")
            return 0.0
    
    def verify_face(self, img1_path, img2_path):
        try:
            emb1 = self.extract_embedding(img1_path)
            emb2 = self.extract_embedding(img2_path)
            
            if emb1 and emb2:
                similarity = self.compare_faces(emb1, emb2)
                return similarity > 0.7, 1 - similarity, similarity
            return False, 1.0, 0.0
        except Exception as e:
            print(f"Error verifying face: {e}")
            return False, 1.0, 0.0
    
    def detect_faces(self, image_path):
        try:
            image = Image.open(image_path)
            return True
        except Exception as e:
            print(f"Error detecting faces: {e}")
            return False
    
    def process_image_from_bytes(self, image_bytes):
        try:
            image = Image.open(io.BytesIO(image_bytes))
            temp_path = "temp_face.jpg"
            image.save(temp_path)
            embedding = self.extract_embedding(temp_path)
            os.remove(temp_path)
            return embedding
        except Exception as e:
            print(f"Error processing image from bytes: {e}")
            return None