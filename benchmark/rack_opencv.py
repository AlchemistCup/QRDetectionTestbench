import cv2
import numpy as np
from typing import List, Optional

qcd = cv2.QRCodeDetector()

def detect(rack_img: cv2.Mat):
    rack = {}
    res, values, _, _ = qcd.detectAndDecodeMulti(rack_img)
    if res:
        for value in values:
            if value:
                value = int(value)
                rack.setdefault(value, 0)
                rack[value] += 1

    return rack


if __name__ == '__main__':
    main()