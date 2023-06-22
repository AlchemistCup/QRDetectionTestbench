import cv2
import numpy as np
from typing import Dict
from rack_reliability_benchmark import full_benchmark
import zxingcpp
import asyncio
import concurrent.futures
from sys import stderr

def detect(rack_img: cv2.Mat) -> Dict[int, int]:
    rack = {}
    barcodes = zxingcpp.read_barcodes(rack_img, binarizer=zxingcpp.Binarizer.GlobalHistogram)
    
    for code in barcodes:
        value = int(code.data)
        rack.setdefault(value, 0)
        rack[value] += 1

    return rack

def detect_all_orientation_multiprocess(pool, rack_img: cv2.Mat) -> Dict[int, int]:
    results = []

    for orientation in [0, 1, 2, 3]:
        rotated_img = np.rot90(rack_img, k=orientation)
        result = pool.apply_async(detect, (rotated_img,))
        results.append(result)
    
    rack_union = {}
    for result in results:
        rack = result.get()
        for key, value in rack.items():
            rack_union[key] = max(rack_union.get(key, 0), value)

    return rack_union

async def detect_all_orientation(executor: concurrent.futures.ProcessPoolExecutor, rack_img: cv2.Mat) -> Dict[int, int]:
    results = []
    loop = asyncio.get_event_loop()

    for orientation in [0, 1, 2, 3]:
        rotated_img = np.rot90(rack_img, k=orientation)
        result = loop.run_in_executor(executor, detect, rotated_img)
        results.append(result)
    
    rack_union = {}
    for rack in await asyncio.gather(*results):
        for key, value in rack.items():
            rack_union[key] = max(rack_union.get(key, 0), value)

    return rack_union
        

def main():
    iterations = 1000
    print('Testing standard library')
    full_benchmark(iterations, detect)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    print('Testing multi-orientation library')
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        full_benchmark(iterations, 
            lambda img:
                loop.run_until_complete(detect_all_orientation(executor, img))
        )

if __name__ == '__main__':
    main()