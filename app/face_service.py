import os
import json
import numpy as np
import cv2
from PIL import Image
import io


class FaceRecognitionService:
    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def _image_to_cv2(self, image_path):
        image = cv2.imread(image_path)
        return image

    def _cv2_to_embedding(self, gray_face):
        resized = cv2.resize(gray_face, (100, 100))
        pixels = resized.flatten().astype(np.float32) / 255.0
        return pixels.tolist()

    def extract_embeddings(self, image_path):
        embeddings = []
        image = self._image_to_cv2(image_path)
        if image is None:
            return embeddings

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            embedding = self._cv2_to_embedding(face_roi)
            embeddings.append({
                "embedding": embedding,
                "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
            })

        return embeddings

    def extract_embedding(self, image_path):
        embeddings = self.extract_embeddings(image_path)
        if embeddings:
            return json.dumps(embeddings[0]["embedding"])
        image = self._image_to_cv2(image_path)
        if image is None:
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (100, 100))
        pixels = resized.flatten().astype(np.float32) / 255.0
        return json.dumps(pixels.tolist())

    def compare_faces(self, embedding1, embedding2):
        try:
            if isinstance(embedding1, str):
                emb1 = np.array(json.loads(embedding1), dtype=np.float32)
            else:
                emb1 = np.array(embedding1, dtype=np.float32)
            if isinstance(embedding2, str):
                emb2 = np.array(json.loads(embedding2), dtype=np.float32)
            else:
                emb2 = np.array(embedding2, dtype=np.float32)

            if len(emb1) != len(emb2):
                return 0.0

            diff_sum = float(np.sum((emb1 - emb2) ** 2))
            similarity = max(0.0, 1.0 - (diff_sum / len(emb1)))
            return float(similarity)
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
            faces = self.extract_embeddings(image_path)
            return len(faces) > 0
        except Exception as e:
            print(f"Error detecting faces: {e}")
            return False

    def count_faces(self, image_path):
        try:
            faces = self.extract_embeddings(image_path)
            return len(faces)
        except Exception as e:
            print(f"Error counting faces: {e}")
            return 0

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
