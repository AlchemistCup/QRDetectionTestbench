import cv2
import numpy as np
from typing import List, Optional
import pyboof as pb

qcd = pb.FactoryFiducial(np.uint8).qrcode()

def detect(rack_img: cv2.Mat):
    rack = {}
    rack_img = pb.ndarray_to_boof(rack_img)
    qcd.detect(rack_img)
    for detection in qcd.detections:
        value = int(detection.message)
        rack.setdefault(value, 0)
        rack[value] += 1

    return rack