import operator
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from insightface.app import FaceAnalysis

cv2
# FIXME: Turn into a class
# ?: Maybe try to implement with strategy pattern
def insightface_detector_factory(name: str = 'buffalo_l',
                                 root: str = '~/.insightface',
                                 allowed_modules: Any | None = None,
                                 ctx_id: Any = 0,
                                 det_tresh: float = 0.5,
                                 det_size: Any = (640, 640)) -> Callable[[np.ndarray], list]:
    app = FaceAnalysis(name=name,
                       root=root,
                       allowed_modules=allowed_modules)

    app.prepare(ctx_id=ctx_id, det_thresh=det_tresh, det_size=det_size)

    def detector(img: np.ndarray) -> list:
        return app.get(img)

    return detector

# ? maybe not needed
def detect_faces(img: np.ndarray,
                 face_detector: Callable[[np.ndarray], list]) -> list:
    return face_detector(img)


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


def similarity_calculator(embedding_1: np.ndarray,
                          embedding_2: np.ndarray,
                          similarity_func: Callable[[
                              np.ndarray, np.ndarray], float] = cosine_similarity
                          ) -> float:
    return similarity_func(embedding_1, embedding_2)


def is_similar(similarity: float,
               threshold: float = 0.6,
               oprator_: Callable[[float, float], bool] = operator.gt) -> bool:
    return oprator_(similarity, threshold)

FILE_PATH = Path(__file__)
IMAGES_PATH = FILE_PATH.parent.parent / 'data/images'
def get_images(path: Path, max_image_amount=1_000)-> list[np.ndarray]:
    result = []
    gen = path.iterdir()
    for _ in range(max_image_amount):
        next_path = next(gen)
        result.append(cv2.imread(next_path.resolve().name))
    return result