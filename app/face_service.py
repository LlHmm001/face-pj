import os
import json
import numpy as np
import cv2
import onnxruntime as ort
from PIL import Image
import io


MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


class FaceRecognitionService:
    def __init__(self):
        det_path = os.path.join(MODEL_DIR, "det_10g.onnx")
        rec_path = os.path.join(MODEL_DIR, "w600k_r50.onnx")

        self.det_session = ort.InferenceSession(det_path, providers=["CPUExecutionProvider"])
        self.rec_session = ort.InferenceSession(rec_path, providers=["CPUExecutionProvider"])

        self.det_input_name = self.det_session.get_inputs()[0].name
        self.rec_input_name = self.rec_session.get_inputs()[0].name

    def _image_to_cv2(self, image_path):
        return cv2.imread(image_path)

    def _detect_faces(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        ratio = 640.0 / max(h, w)
        new_h, new_w = int(h * ratio), int(w * ratio)
        img_resized = cv2.resize(img, (new_w, new_h))

        canvas = np.zeros((640, 640, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = img_resized

        blob = (canvas.astype(np.float32) - 127.5) / 128.0
        blob = np.expand_dims(np.transpose(blob, (2, 0, 1)), axis=0)

        outputs = self.det_session.run(None, {self.det_input_name: blob})
        detections = outputs[0]

        faces = []
        scale_h = h / 640.0
        scale_w = w / 640.0

        for det in detections:
            score = float(det[4])
            if score < 0.5:
                continue
            x1 = int(det[0] * scale_w)
            y1 = int(det[1] * scale_h)
            x2 = int(det[2] * scale_w)
            y2 = int(det[3] * scale_h)
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            faces.append({
                "bbox": (x1, y1, x2, y2),
                "score": score
            })

        return faces

    def _rec_embedding(self, face_img):
        face_resized = cv2.resize(face_img, (112, 112))
        blob = (face_resized.astype(np.float32) - 127.5) / 128.0
        blob = np.expand_dims(np.transpose(blob, (2, 0, 1)), axis=0)

        embedding = self.rec_session.run(None, {self.rec_input_name: blob})[0]
        vec = embedding.flatten().astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def extract_embeddings(self, image_path):
        embeddings = []
        image = self._image_to_cv2(image_path)
        if image is None:
            return embeddings

        faces = self._detect_faces(image)
        for face in faces:
            x1, y1, x2, y2 = face["bbox"]
            face_roi = image[y1:y2, x1:x2]
            embedding = self._rec_embedding(face_roi)
            embeddings.append({
                "embedding": embedding,
                "bbox": {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}
            })

        return embeddings

    def extract_embedding(self, image_path):
        embeddings = self.extract_embeddings(image_path)
        if embeddings:
            return json.dumps(embeddings[0]["embedding"])
        image = self._image_to_cv2(image_path)
        if image is None:
            return None
        h, w = image.shape[:2]
        face = image[:h, :w]
        embedding = self._rec_embedding(face)
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
            return max(0.0, min(1.0, dot))
        except Exception as e:
            print(f"Error comparing faces: {e}")
            return 0.0

    def verify_face(self, img1_path, img2_path):
        try:
            emb1 = self.extract_embedding(img1_path)
            emb2 = self.extract_embedding(img2_path)
            if emb1 and emb2:
                similarity = self.compare_faces(emb1, emb2)
                return similarity > 0.4, 1 - similarity, similarity
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
