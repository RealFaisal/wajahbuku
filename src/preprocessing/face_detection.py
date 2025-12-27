from typing import Optional
from PIL import Image
import numpy as np

try:
    from facenet_pytorch import MTCNN
    _HAS_MTCNN = True
except Exception:
    _HAS_MTCNN = False

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

def detect_and_crop_face(pil_image: Image.Image, margin: float = 0.2, target_size: int = 224) -> Optional[Image.Image]:
    """
    Detects the largest face and returns a cropped PIL.Image resized to (target_size, target_size).
    - margin: fraction of box to expand (e.g., 0.2 => 20% expand)
    - Returns None if no face found.
    """
    if _HAS_MTCNN:
        try:
            mtcnn = MTCNN(keep_all=False)
            face = mtcnn(pil_image)
            # mtcnn returns a tensor if face found, else None
            if face is None:
                return None
            # convert tensor -> PIL
            face_pil = Image.fromarray((face.permute(1, 2, 0).numpy() * 255).astype('uint8'))
            face_pil = face_pil.resize((target_size, target_size))
            return face_pil
        except Exception:
            # fallthrough to cv2
            pass

    if _HAS_CV2:
        try:
            img = np.array(pil_image.convert("RGB"))
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            rects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if len(rects) == 0:
                return None
            # choose largest face
            rects = sorted(rects, key=lambda r: r[2]*r[3], reverse=True)
            x, y, w, h = rects[0]
            # expand box by margin
            mx = int(w * margin)
            my = int(h * margin)
            x1 = max(0, x - mx)
            y1 = max(0, y - my)
            x2 = min(img.shape[1], x + w + mx)
            y2 = min(img.shape[0], y + h + my)
            face = img[y1:y2, x1:x2, :]
            face_pil = Image.fromarray(face).resize((target_size, target_size))
            return face_pil
        except Exception:
            return None

    # if no detector available, just resize full image (not ideal)
    try:
        return pil_image.convert("RGB").resize((target_size, target_size))
    except Exception:
        return None
