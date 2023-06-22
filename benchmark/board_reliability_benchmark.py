import numpy as np
from time import time
from typing import Callable
from pathlib import Path

from generate_board import BOARD_DIMS, generate_board
from generate_artefacts import *

BAD_BOARD_DIR = Path(__name__).resolve().parent / 'bad_boards'
BAD_BOARD_RAW_DIR = Path(__name__).resolve().parent / 'bad_boards_raw'
i = 1

# Replicates image preprocessing for sensor pipeline
def preprocess_image(img: cv2.Mat) -> cv2.Mat:
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_grey, (5, 5), 1) # Remove gaussian noise
    img_threshold = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=11, C=2) # Potentially tweek params
    return img_threshold

def find_differences(result, reference):
    diff_idx = np.where(result != reference)
    row_idx, col_idx = diff_idx

    differences = [
        (res, ref, (row, col))
        for res, ref, row, col in zip(result[diff_idx], reference[diff_idx], row_idx, col_idx)
    ]

    return differences

def calculate_accuracy(errors):
    total = BOARD_DIMS[1] * BOARD_DIMS[0]
    return (total - errors) / total * 100

def benchmark(iterations: int, frac_empty: float, detect_board: Callable):
    global i
    times = []
    n_of_errors = []
    for _ in range(iterations):
        grid, ref = generate_board(frac_empty)
        img = artefact_preprocess(grid)
        img = randomly_add_artefacts(img)

        start = time()
        grid = preprocess_image(img)
        res = detect_board(grid)
        end = time()

        times.append((end - start) * 1000)
        if not (res == ref).all():
            errors = find_differences(res, ref)
            n_of_errors.append(len(errors))
            name = f'{i:03}_{len(errors)}.png'
            cv2.imwrite(f'{BAD_BOARD_RAW_DIR / name}', img)
            for _, _, (row, col) in errors:
                assert img.shape[0] == img.shape[1]
                square_len = img.shape[0] // 8
                x = col * square_len
                y = row * square_len
                img = cv2.rectangle(img, (x+8, y+8), (x+square_len-8, y+square_len-8), color=(0, 0, 255), thickness=2)
            cv2.imwrite(f'{BAD_BOARD_DIR / name}', img)
            i += 1
        else:
           n_of_errors.append(0)

    times = np.array(times)
    n_of_errors = np.array(n_of_errors)
    accuracy = np.vectorize(calculate_accuracy)(n_of_errors)

    print(f"Benchmark complete ({iterations} iterations): ")
    print(f"\tTime: mean = {times.mean():.5f}ms, std. dev = {times.std():.5f}ms")
    print(f"\tAccuracy: mean = {accuracy.mean():.5f}%, std. dev = {accuracy.std():.5f}%")

def full_benchmark(iterations: int, detect_grid: Callable):
    for frac_tiles in [0., 0.1, 0.25, 0.5, 0.75, 0.9]:
        print(f"Testing {frac_tiles * 100}% tiles")
        benchmark(iterations, 1 - frac_tiles, detect_grid)
