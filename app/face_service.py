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
        self.win_size = (64, 64)
        self.hog = cv2.HOGDescriptor(
            _winSize=self.win_size,
            _blockSize=(16, 16),
            _blockStride=(8, 8),
            _cellSize=(8, 8),
            _nbins=9
        )

    def _image_to_cv2(self, image_path):
        return cv2.imread(image_path)

    def _hog_embedding(self, face):
        resized = cv2.resize(face, self.win_size)
        hog_features = self.hog.compute(resized)
        vec = hog_features.flatten().astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

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
            embedding = self._hog_embedding(face_roi)
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
        embedding = self._hog_embedding(gray)
        return json.dumps(embedding)

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

            dot = float(np.dot(emb1, emb2))
            similarity = (dot + 1.0) / 2.0
            return max(0.0, min(1.0, similarity))
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
