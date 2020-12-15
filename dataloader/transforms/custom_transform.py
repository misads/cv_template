import os
import cv2
import numpy as np


def read_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f'{image_path} not found.')

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image /= 255.0  # 转成0~1之间

    return image