import cv2
import numpy as np
import segno
from pyzxing import BarCodeReader
from time import time
from random import randint

TEST_FILE = 'qr_test.png'
START = 57004
ATTEMPTS = 100000 - 57004

total = 0
reader = BarCodeReader()
undetected_codes = []
undecoded_codes = []

for i in range(START, START+ATTEMPTS):
    x = i
    qrcode = segno.make(x, version=1)
    assert qrcode.error == 'H'
    qrcode.save(TEST_FILE, scale = 5)

    img = cv2.imread(TEST_FILE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (88, 88))

    start = time()
    barcodes = reader.decode_array(img)
    end = time()
    if len(barcodes) == 0:
        print(f"Unable to detect {x}")
        undetected_codes.append(x)
    else:
        assert len(barcodes) == 1, f"Detected {len(barcodes)} qr codes (input = {x})"
        if 'raw' not in barcodes[0]:
            print(f"Unable to decode {x}")
            undecoded_codes.append(x)
        else:
            assert barcodes[0]['raw'].decode('utf-8') == str(x), f"Mismatch between detected value {barcodes[0]['raw'].decode('utf-8')} and input {str(x)}"
    total += (end - start) * 1000

print(f'Took an average of {(total / ATTEMPTS):.2f}ms for {ATTEMPTS} attempts')
print(f"Unable to detect the following codes:\n{undetected_codes}")
print(f"Unable to decode the following codes:\n{undecoded_codes}")