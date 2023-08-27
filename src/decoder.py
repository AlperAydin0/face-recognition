import base64
import numpy as np


def base64_decoder(img: bytes, img_shape: tuple[int, int, int]) -> np.ndarray:
    # decodes and converts given img encoded as base64 to np.ndarray
    return np.frombuffer(base64.b64decode(img), dtype=np.uint8).reshape(img_shape)
