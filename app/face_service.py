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

        det_outputs = self.det_session.run(None, {self.det_input_name: blob})

        stride_configs = [
            (8, 6400, 0, 3),
            (16, 1600, 1, 4),
            (32, 400, 2, 5),
        ]

        all_boxes = []
        all_scores = []
        scale_h = h / 640.0
        scale_w = w / 640.0

        for stride, num_cells, score_idx, bbox_idx in stride_configs:
            scores = det_outputs[score_idx].reshape(num_cells, 2)
            bboxes = det_outputs[bbox_idx].reshape(num_cells, 2, 4)

            grid_size = 640 // stride
            for cell in range(num_cells):
                for anchor in range(2):
                    s = float(scores[cell, anchor])
                    if s < 0.5:
                        continue

                    grid_y = cell // grid_size
                    grid_x = cell % grid_size
                    cx = (grid_x + 0.5) * stride
                    cy = (grid_y + 0.5) * stride

                    dx1 = float(bboxes[cell, anchor, 0])
                    dy1 = float(bboxes[cell, anchor, 1])
                    dx2 = float(bboxes[cell, anchor, 2])
                    dy2 = float(bboxes[cell, anchor, 3])

                    x1 = (cx - dx1 * stride) * scale_w
                    y1 = (cy - dy1 * stride) * scale_h
                    x2 = (cx + dx2 * stride) * scale_w
                    y2 = (cy + dy2 * stride) * scale_h

                    x1 = max(0, min(w - 1, int(x1)))
                    y1 = max(0, min(h - 1, int(y1)))
                    x2 = max(0, min(w - 1, int(x2)))
                    y2 = max(0, min(h - 1, int(y2)))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    all_boxes.append([x1, y1, x2, y2])
                    all_scores.append(s)

        if not all_boxes:
            return []

        indices = cv2.dnn.NMSBoxes(all_boxes, all_scores, 0.5, 0.4)

        faces = []
        if len(indices) > 0:
            for idx in indices.flatten():
                x1, y1, x2, y2 = all_boxes[idx]
                faces.append({
                    "bbox": (x1, y1, x2, y2),
                    "score": all_scores[idx]
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
