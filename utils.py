import base64
import json

import numpy as np
import cv2.cv2 as cv2


def array_to_json(items):
    array_of_jsons = [json.dumps(item.__dict__) for item in items]
    return "[" + ", ".join(array_of_jsons) + "]"


def image_from_base64_encoded_bytes(data):
    image_bytes = base64.b64decode(data)
    np_data = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(np_data, cv2.IMREAD_COLOR)
