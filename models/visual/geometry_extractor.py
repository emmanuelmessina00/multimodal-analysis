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
            # Eye Aspect Ratio (EAR): rapporto medio tra l'altezza verticale delle palpebre
            # e la larghezza orizzontale dell'occhio (mediato tra occhio destro e sinistro).
            "ear": self._compute_ear(pts),

            # Mouth Aspect Ratio (MAR): rapporto tra l'apertura verticale interna delle labbra
            "mar": self._compute_mar(pts),

            # Elevazione normalizzata delle sopracciglia: distanza media verticale tra il centro
            # del sopracciglio e il margine superiore dell'occhio, divisa per la distanza
            # inter-oculare.
            "brow_raise": self._compute_brow_raise(pts),

            # Angolo di Pitch (in gradi): inclinazione della testa sull'asse orizzontale
            # (annuire, guardare in basso verso il tabellone o alzare lo sguardo).
            "pitch": pitch,

            # Angolo di Yaw (in gradi): rotazione orizzontale della testa a destra o a sinistra
            # (girarsi a guardare gli altri giocatori intorno al tavolo).
            "yaw": yaw,

            # Angolo di Roll (in gradi): inclinazione laterale della testa verso una spalla
            # (tipico gesto di perplessità, dubbio o riflessione durante il turno di gioco).
            "roll": roll,

            # Blendshape Sorriso (in [0, 1]): media dell'attivazione muscolare degli angoli
            # della bocca verso l'alto (mouthSmileLeft/Right, analogo all'AU12 di Ekman).
            "bs_smile": (bs.get("mouthSmileLeft", 0.0) + bs.get("mouthSmileRight", 0.0)) / 2.0,

            # Blendshape Apertura Mandibola (in [0, 1]): grado di abbassamento della mandibola
            # stimato dal modello 3D (complementare a MAR, più robusto alle rotazioni).
            "bs_jaw_open": bs.get("jawOpen", 0.0),

            # Blendshape Sollevamento Sopracciglio Interno (in [0, 1]): attivazione della parte
            # mediale delle sopracciglia (analogo all'AU1 di Ekman, legato a sorpresa o apprensione).
            "bs_brow_up": bs.get("browInnerUp", 0.0),

            # Blendshape Chiusura Palpebre (in [0, 1]): media di eyeBlinkLeft/Right; misura
            # direttamente la chiusura dell'occhio anche quando la testa è inclinata.
            "bs_blink": (bs.get("eyeBlinkLeft", 0.0) + bs.get("eyeBlinkRight", 0.0)) / 2.0,
        }

    def close(self):
        if self.detector:
            self.detector.close()

    def close(self):
        if self.detector:
            self.detector.close()


if __name__ == "__main__":
    ## Un test giusto per vedere se prende bene le features facciali (scommentare)
    """
    import glob
    from utils.video_utils import natural_sort_key

    video_files = sorted(
        [p for p in glob.glob(os.path.join(PROJECT_ROOT, "data", "video", "*")) if p.lower().endswith(".mp4")],
        key=natural_sort_key,
    )
    print("Video ordinati:", [os.path.basename(v) for v in video_files])

    cap = cv2.VideoCapture(video_files[0])
    cap.set(cv2.CAP_PROP_POS_FRAMES, 4800)  # keyframe reale della window_id = 7

    extractor = FacialGeometryExtractor()
    frame_idx = 0
    step = 4

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1
        if frame_idx % step != 0:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        res = extractor.detector.detect(mp_image)

        vis_rgb = extractor.draw_landmarks_on_image(rgb_frame, res)
        vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)

        feats = extractor.extract_frame(frame)
        if feats is not None:
            lines = [
                f"EAR: {feats['ear']:.3f} | MAR: {feats['mar']:.3f} | Brow: {feats['brow_raise']:.3f}",
                f"Pitch: {feats['pitch']:+.1f} | Yaw: {feats['yaw']:+.1f} | Roll: {feats['roll']:+.1f}",
                f"Smile: {feats['bs_smile']:.2f} | JawOpen: {feats['bs_jaw_open']:.2f} | Blink: {feats['bs_blink']:.2f}",
            ]
            for i, text in enumerate(lines):
                y_pos = 30 + i * 28
                cv2.putText(vis_bgr, text, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(vis_bgr, text, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 180), 2, cv2.LINE_AA)

        h, w = vis_bgr.shape[:2]
        max_h = 850
        if h > max_h:
            scale = max_h / h
            vis_bgr = cv2.resize(vis_bgr, (int(w * scale), int(h * scale)))

        cv2.imshow("MediaPipe FaceLandmarker - Sanity Check", vis_bgr)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            cv2.waitKey(0)

    cap.release()
    extractor.close()
    cv2.destroyAllWindows()
    
    """
    pass