import cv2
import numpy as np
import segno
import zxingcpp
from time import time
from random import randint

TEST_FILE = 'qr_micro_test.png'
START = 0
ATTEMPTS = 100000

total = 0
bad_codes = []

for i in range(START, START+ATTEMPTS):
    x = i
    qrcode = segno.make(x, version='M4')
    assert qrcode.error == 'Q'
    qrcode.save(TEST_FILE, scale = 5)

    img = cv2.imread(TEST_FILE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (88, 88))

    start = time()
    barcodes = zxingcpp.read_barcodes(img, binarizer=zxingcpp.Binarizer.GlobalHistogram)
    end = time()
    if len(barcodes) == 0:
        # print(f"Unable to detect {x}")
        bad_codes.append(x)
    else:
        assert len(barcodes) == 1, f"Detected {len(barcodes)} qr codes (input = {x})"
        assert barcodes[0].text == str(x), f"Mismatch between detected value {barcodes[0].text} and input {str(x)}"
    total += (end - start) * 1000

print(f'Took an average of {(total / ATTEMPTS):.2f}ms for {ATTEMPTS} attempts')
print(f"Unable to detect the following codes:\n{bad_codes}")