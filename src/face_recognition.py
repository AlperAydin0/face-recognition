from dataclasses import dataclass
import operator
from pathlib import Path
from typing import Callable, Protocol

import cv2
import numpy as np
from numpy import ndarray
from insightface.model_zoo import RetinaFace, ArcFaceONNX

ARCFACE_PATH = '/home/alper/.insightface/models/buffalo_l/w600k_r50.onnx'
RETINAFACE_PATH = '/home/alper/.insightface/models/buffalo_l/det_10g.onnx'

# def insightface_detector_factory(name: str = 'buffalo_l',
#                                  root: str = '~/.insightface',
#                                  allowed_modules: Any | None = None,
#                                  ctx_id: Any = 0,
#                                  det_tresh: float = 0.5,
#                                  det_size: Any = (640, 640)) -> Callable[[np.ndarray], list]:
#     app = FaceAnalysis(name=name,
#                        root=root,
#                        allowed_modules=allowed_modules)

#     app.prepare(ctx_id=ctx_id, det_thresh=det_tresh, det_size=det_size)

#     def detector(img: np.ndarray) -> list:
#         return app.get(img)

#     return detector


class FaceNotDetectedExeption(Exception):
    def __init__(self, *args: object,message: str = 'Image doesn\'t contain a face') -> None:
        super().__init__(message, *args)


@dataclass
class Face:
    bbox: ndarray
    confidance: float
    kps: ndarray


class Detector(Protocol):
    def detect(self, img: ndarray) -> list[Face] | None:
        ...


class RetinaFaceDetector(Detector):
    def __init__(self,
                 retinaface_path: Path | str = RETINAFACE_PATH,
                 input_size: tuple[int, int] = (640, 640)) -> None:
        self.input_size = input_size
        self.detector = RetinaFace(retinaface_path)

    def detect(self, img: ndarray) -> list[Face] | None:
        res = self.detector.detect(img, input_size=self.input_size)
        if res[0].shape[0] == 0:
            return None
        return [Face(bbox[:-1], bbox[-1], kpss) for bbox, kpss in zip(res[0], res[1])] # type: ignore


class Recognizer(Protocol):
    def get_embedding(self, img: ndarray, face: Face) -> ndarray:
        ...


class ArcFaceRecognizer(Recognizer):
    def __init__(self,
                 arcface_path: Path | str = ARCFACE_PATH) -> None:
        self.recognizer = ArcFaceONNX(arcface_path)

    def get_embedding(self, img: ndarray, face: Face) -> ndarray:
        return self.recognizer.get(img, face)


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


def get_images(max_image_amount=1_000) -> list[np.ndarray]:
    file_path = Path(__file__)
    images_path = file_path.parent.parent / 'data/images'
    result = []
    gen = images_path.iterdir()
    for _ in range(max_image_amount):
        next_path = next(gen)
        result.append(cv2.imread(next_path.resolve().name))
    return result
