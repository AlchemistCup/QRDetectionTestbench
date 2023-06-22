import cv2
import numpy as np
import segno
from pyzbar import pyzbar
from time import time
from random import randint

TEST_FILE = 'qr_test.png'
START = 0
ATTEMPTS = 100000

total = 0

for i in range(START, START+ATTEMPTS):
    x = randint(0, 10000)
    qrcode = segno.make(x, version=1)
    assert qrcode.error == 'H'
    qrcode.save(TEST_FILE, scale = 5)

    img = cv2.imread(TEST_FILE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (88, 88))

    start = time()
    barcodes = pyzbar.decode(img)
    end = time()
    assert len(barcodes) == 1, f"Detected {len(barcodes)} qr codes (input = {x})"
    assert barcodes[0].data.decode("utf-8") == str(x)
    total += (end - start) * 1000

print(f'Took an average of {(total / ATTEMPTS):.2f}ms for {ATTEMPTS} attempts')
