import numpy as np

def base64_decoder(img: bytes, img_shape: tuple[int, int, int]) -> np.ndarray:
    # decodes and converts given img encoded as base64 to np.ndarray
    result_arr = np.frombuffer(img, dtype=np.uint8)
    final_arr = result_arr.reshape(img_shape)
    return final_arr
