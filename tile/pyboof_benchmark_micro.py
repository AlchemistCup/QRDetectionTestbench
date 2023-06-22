import cv2
import numpy as np
import pyboof as pb
import segno
from time import time
from random import randint
# uninstall Java JDK after testing
TEST_FILE = 'qr_micro_test.png'
START = 0
ATTEMPTS = 100000

qr_pyboof = pb.FactoryFiducial(np.uint8).microqr()
total_pyboof = 0

bad_codes = []

generator = pb.MicroQrCodeGenerator(pixels_per_module=5)
#generator.set_version('M4')
generator.set_error('Q')


for i in range(START, START+ATTEMPTS):
    x = i
    generator.set_message(x)
    qrcode = generator.generate()
    img = pb.boof_to_ndarray(qrcode)

    #assert qrcode.error == 'Q'
    #qrcode.save(TEST_FILE, scale = 5)

    #img = cv2.imread(TEST_FILE)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (88, 88))
    print(img.shape)
    cv2.imwrite('test.png', img)
    exit(0)

    start = time()
    image = pb.ndarray_to_boof(img)
    qr_pyboof.detect(image)
    end = time()
    if len(qr_pyboof.detections) == 0:
        print(f"Unable to detect {x}")
        bad_codes.append(x)
    else:
        assert len(qr_pyboof.detections) == 1, f"Detected {len(qr_pyboof.detections)} qr codes (input = {x})"
        assert qr_pyboof.detections[0].message == str(x)

    total_pyboof += (end - start) * 1000

print(f'Took an average of {(total_pyboof / ATTEMPTS):.2f}ms for {ATTEMPTS} attempts')
print(f"Unable to detect the following codes:\n{bad_codes}")