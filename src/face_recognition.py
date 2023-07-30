from typing import Any, Callable

import numpy as np
from insightface.app import FaceAnalysis


def insightface_detector_generator(name: str = 'buffalo_l',
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


def detect_faces(img: np.ndarray, face_detector: Callable[[np.ndarray], list]) -> list:
    return face_detector(img)


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    ...


def similarity_calculator(embedding_1: np.ndarray,
                          embedding_2: np.ndarray,
                          similarity_func: Callable[[
                              np.ndarray, np.ndarray], float] = cosine_similarity
                          ) -> float:
    return similarity_func(embedding_1, embedding_2)


def is_similar(similarity: float, treshold: float = 0.6) -> bool:
    ...
