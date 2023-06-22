import cv2
import numpy as np
import segno
from time import time
from random import randint

TEST_FILE = 'qr_test.png'
ATTEMPTS = 100000
START = 0

qr_opencv = cv2.QRCodeDetector()
total_opencv = 0
bad_codes = []

for i in range(START, START+ATTEMPTS):
    x = i
    qrcode = segno.make(x, version=1)
    assert qrcode.error == 'H'
    qrcode.save(TEST_FILE, scale = 5)

    img = cv2.imread(TEST_FILE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (88, 88))

    start = time()
    detected, points, straight_qrcode = qr_opencv.detectAndDecode(img)
    end = time()
    if not detected:
        print(f"Unable to detect {x}")
        bad_codes.append(x)
    else:
        assert str(detected) == str(x), f"Mismatch between detected value {detected} and input {x}"
    total_opencv += (end - start) * 1000

print(f'Took an average of {(total_opencv / ATTEMPTS):.2f}ms for {ATTEMPTS} attempts')
print(f"Unable to detect the following codes:\n{bad_codes}")