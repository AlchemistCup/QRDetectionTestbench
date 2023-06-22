import numpy as np
import cv2
import random
from typing import List, Callable

def artefact_preprocess(img, offset=50):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img[img < (255/2 - offset)] += offset
    img[img > (255/2 + offset)] -= offset 
    return img

def add_shot_noise(img):
    return np.random.poisson(lam=img, size=None).astype(np.uint8)

def adjust_brightness(img, range=(-50, 50)):
    brightness = random.randint(*range)
    return cv2.convertScaleAbs(img, alpha=1, beta=brightness)

def add_glare(img, intensity_range=(0.1, 0.4), size_range=(0.2, 0.8), n_range=(0, 3)):
    img = img.astype(np.float32) / 255.0
    
    height, width, _ = img.shape
    glare_mask_combined = np.zeros((height, width), dtype=np.float32)

    glare_positions = []
    glare_radii = []
    n = random.randint(*n_range)
    for _ in range(n):
        glare_mask = np.zeros((height, width), dtype=np.float32)

        valid_position = False
        while not valid_position:
            center_x, center_y = np.random.randint(0, width), np.random.randint(0, height)
            glare_radius = np.random.uniform(*size_range) * min(width, height)

            overlap = False
            for existing_position, existing_radius in zip(glare_positions, glare_radii):
                distance = np.sqrt((center_x - existing_position[0]) ** 2 + (center_y - existing_position[1]) ** 2)
                if distance < glare_radius + existing_radius:
                    overlap = True
                    break

            if not overlap:
                valid_position = True

        glare_intensity_random = np.random.uniform(*intensity_range)

        cv2.circle(glare_mask, (center_x, center_y), int(glare_radius), (1.0, 1.0, 1.0), -1, cv2.LINE_AA)
        glare_mask_combined += glare_mask * glare_intensity_random

        glare_positions.append((center_x, center_y))
        glare_radii.append(glare_radius)

    glare_effect = img + glare_mask_combined[..., np.newaxis]
    glare_effect = np.clip(glare_effect, 0.0, 1.0)
    glare_effect = (glare_effect * 255.0).astype(np.uint8)

    return glare_effect

def randomly_add_artefacts(img, available_artefacts: List[Callable] = [add_shot_noise, adjust_brightness, add_glare]):
    n = random.randint(1, len(available_artefacts))
    selected = random.sample(available_artefacts, n)
    selected.sort(key=lambda func: available_artefacts.index(func))
    for apply_artefact in selected:
        img = apply_artefact(img)

    return img