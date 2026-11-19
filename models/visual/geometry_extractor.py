import os
import sys

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

class FacialGeometryExtractor:
    def __init__(self):
        self.base_options = python.BaseOptions(model_asset_path=os.path.join(PROJECT_ROOT, "checkpoints", "face_landmarker.task"))
        self.options = vision.FaceLandmarkerOptions(base_options=self.base_options,
                                               output_face_blendshapes=True,
                                               output_facial_transformation_matrixes=True,
                                               num_faces=1)
        self.detector= vision.FaceLandmarker.create_from_options(self.options)

    @staticmethod
    def draw_landmarks_on_image(rgb_image, detection_result):
      face_landmarks_list = detection_result.face_landmarks
      annotated_image = np.copy(rgb_image)

      # Loop through the detected faces to visualize.
      for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.


        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style())
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style())
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
              landmark_drawing_spec=None,
              connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style())
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
              landmark_drawing_spec=None,
              connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style())

      return annotated_image

    @staticmethod
    def _compute_ear(pts: np.ndarray) -> float:
        def eye_ratio(idx: list[int]) -> float:
            p1, p2, p3, p4, p5, p6 = pts[idx]
            vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
            horizontal = 2.0 * np.linalg.norm(p1 - p4)
            return float(vertical / (horizontal + 1e-6))

        left_ear = eye_ratio([33, 160, 158, 133, 153, 144])
        right_ear = eye_ratio([362, 385, 387, 263, 373, 380])
        return (left_ear + right_ear) / 2.0

    @staticmethod
    def _compute_mar(pts: np.ndarray) -> float:
        vertical = np.linalg.norm(pts[13] - pts[14])
        horizontal = np.linalg.norm(pts[78] - pts[308])
        return float(vertical / (horizontal + 1e-6))

    @staticmethod
    def _compute_brow_raise(pts: np.ndarray) -> float:
        inter_ocular = np.linalg.norm(pts[33] - pts[263]) + 1e-6
        left_dist = np.linalg.norm(pts[105] - pts[159])
        right_dist = np.linalg.norm(pts[334] - pts[386])
        return float(((left_dist + right_dist) / 2.0) / inter_ocular)

    @staticmethod
    def _matrix_to_euler(mat_4x4: np.ndarray) -> tuple[float, float, float]:
        R = mat_4x4[:3, :3]
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        pitch = float(np.degrees(np.arctan2(R[2, 1], R[2, 2])))
        yaw = float(np.degrees(np.arctan2(-R[2, 0], sy)))
        roll = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
        return pitch, yaw, roll

    def extract_frame(self, bgr_frame: np.ndarray) -> dict[str, float] | None:
        if bgr_frame is None:
            raise ValueError("Could not decode image")
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        res=self.detector.detect(mp_image)

        if not res.face_landmarks:
            return None

        h, w = bgr_frame.shape[:2]
        pts = np.array([[lm.x * w, lm.y * h] for lm in res.face_landmarks[0]], dtype=np.float32)

        pitch, yaw, roll = (0.0, 0.0, 0.0)
        if res.facial_transformation_matrixes:
            pitch, yaw, roll = self._matrix_to_euler(res.facial_transformation_matrixes[0])

        bs = {}
        if res.face_blendshapes:
            bs = {b.category_name: float(b.score) for b in res.face_blendshapes[0]}

        return {
            "ear": self._compute_ear(pts),
            "mar": self._compute_mar(pts),
            "brow_raise": self._compute_brow_raise(pts),
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
            "bs_smile": (bs.get("mouthSmileLeft", 0.0) + bs.get("mouthSmileRight", 0.0)) / 2.0,
            "bs_jaw_open": bs.get("jawOpen", 0.0),
            "bs_brow_up": bs.get("browInnerUp", 0.0),
            "bs_blink": (bs.get("eyeBlinkLeft", 0.0) + bs.get("eyeBlinkRight", 0.0)) / 2.0,
        }

    def close(self):
        if self.detector:
            self.detector.close()

    def close(self):
        if self.detector:
            self.detector.close()


if __name__ == "__main__":
    import glob

    video_files = sorted(glob.glob(os.path.join(PROJECT_ROOT, "data", "video", "*.mp4")))
    if not video_files:
        raise FileNotFoundError("Nessun file .mp4 trovato in data/video")

    cap = cv2.VideoCapture(video_files[0])
    cap.set(cv2.CAP_PROP_POS_FRAMES, 150)  # frame di prova a ~5 secondi
    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError("Impossibile leggere il frame di test.")

    extractor = FacialGeometryExtractor()
    feats = extractor.extract_frame(frame)
    extractor.close()

    print("Feature geometriche estratte sul frame 150:")
    for k, v in (feats or {}).items():
        print(f"  {k}: {v:.4f}")