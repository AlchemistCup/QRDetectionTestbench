import numpy as np
from time import time
from typing import Dict, Callable
from pathlib import Path
import cv2

from generate_rack import generate_rack
from generate_artefacts import *

BAD_RACK_DIR = Path(__name__).resolve().parent / 'bad_racks'
BAD_RACK_RAW_DIR = Path(__name__).resolve().parent / 'bad_racks_raw'

error_id = 1

# Replicates image preprocessing for sensor pipeline
def preprocess_image(img: cv2.Mat) -> cv2.Mat:
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_grey, (5, 5), 1) # Remove gaussian noise
    img_threshold = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=11, C=4) # Potentially tweek params
    return img_threshold

def calculate_accuracy_metrics(result: Dict[int, int], reference: Dict[int, int]):
    tp = 0
    fp = 0
    fn = 0

    for key in reference:
        if key in result:
            tp += min(result[key], reference[key])
            fp += max(result[key] - reference[key], 0)
            fn += max(reference[key] - result[key], 0)
        else:
            fn += reference[key]

    for key in result:
        if key not in reference:
            fp += result[key]

    precision = tp / (tp + fp) if tp + fp > 0 else 1.
    recall = tp / (tp + fn) if tp + fn > 0 else 1.
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return f1, precision, recall

def benchmark(iterations: int, n_of_tiles: int, detect_rack: Callable[[cv2.Mat], Dict[int, int]]):
    global error_id
    times = np.empty((iterations,), dtype=np.float32)
    f1_scores = np.empty((iterations,), dtype=np.float32)
    precisions = np.empty((iterations,), dtype=np.float32)
    recalls = np.empty((iterations,), dtype=np.float32)
    for i in range(iterations):
        rack, ref = generate_rack(n_of_tiles)
        img = artefact_preprocess(rack)
        img = randomly_add_artefacts(img)
        
        start = time()
        rack = preprocess_image(img)
        res = detect_rack(rack)
        end = time()

        if res is None:
            cv2.imwrite('orientation_error_raw.png', img)
            exit(0)

        times[i] = (end - start) * 1000
        f1_scores[i], precisions[i], recalls[i] = calculate_accuracy_metrics(res, ref)

        if f1_scores[i] < 0.9999:
            name = f'{error_id:03}_{int(f1_scores[i] * 10000)}.png'
            cv2.imwrite(f'{BAD_RACK_DIR / name}', rack)
            cv2.imwrite(f'{BAD_RACK_RAW_DIR / name}', img)
            error_id += 1
    
    print(f"Benchmark complete ({iterations} iterations): ")
    print(f"\tTime: mean = {times.mean():.5f}ms, std. dev = {times.std():.5f}ms")
    print(f"\tF1 Score: mean = {f1_scores.mean():.5f}, std. dev = {f1_scores.std():.5f}")
    print(f"\tPrecision: mean = {precisions.mean():.5f}, std. dev = {precisions.std():.5f}")
    print(f"\tRecall: mean = {recalls.mean():.5f}, std. dev = {recalls.std():.5f}")



def full_benchmark(iterations: int, detect_rack: Callable[[cv2.Mat], Dict[int, int]]):
    for n_of_tiles in range(8):
        print(f"Testing {n_of_tiles} tiles")
        benchmark(iterations, n_of_tiles, detect_rack)

def main():
    iterations = 1000
    # print('Testing OpenCV library')
    # full_benchmark(iterations, opencv.detect)

    # print('Testing BoofCV library')
    # full_benchmark(iterations, boofcv.detect)

    # print('Testing ZBar library')
    # full_benchmark(iterations, zbar.detect)

if __name__ == '__main__':
    main()
