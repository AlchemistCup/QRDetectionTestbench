import numpy as np
from pathlib import Path
import re
import json

def parse_data_to_dict(data):
    result = {}
    method_pattern = r"Testing (.+?) library"
    tile_pattern = r"Testing (\d+) tiles"
    time_pattern = r"Time: mean = (\d+\.?\d*)ms, std. dev = (\d+\.?\d*)ms"
    accuracy_pattern = r"F1 Score: mean = (\d+\.?\d*), std. dev = (\d+\.?\d*)"
    
    current_method = None
    
    lines = data.strip().split("\n")
    
    for line in lines:
        method_match = re.search(method_pattern, line)
        if method_match:
            method_name = method_match.group(1)
            result[method_name] = {}
            current_method = method_name
            continue
        
        tile_match = re.search(tile_pattern, line)
        if tile_match and current_method:
            tile_fraction = float(tile_match.group(1))
            if current_method not in result:
                result[current_method] = {}
            result[current_method][tile_fraction] = {}
            continue
        
        time_match = re.search(time_pattern, line)
        if time_match and current_method:
            time_mean = float(time_match.group(1))
            time_std = float(time_match.group(2))
            if tile_fraction in result[current_method]:
                result[current_method][tile_fraction]['time_mean'] = time_mean
                result[current_method][tile_fraction]['time_std'] = time_std
            continue
        
        accuracy_match = re.search(accuracy_pattern, line)
        if accuracy_match and current_method:
            accuracy_mean = float(accuracy_match.group(1))
            accuracy_std = float(accuracy_match.group(2))
            if tile_fraction in result[current_method]:
                result[current_method][tile_fraction]['accuracy_mean'] = accuracy_mean
                result[current_method][tile_fraction]['accuracy_std'] = accuracy_std
            continue
    
    return result

import numpy as np

def convert_to_vectors(parsed_data):
    method_names = list(parsed_data.keys())
    vectors = {}
    
    for method_name in method_names:
        method_data = parsed_data[method_name]
        tile_fractions = np.array(list(method_data.keys()))
        
        time_means = np.array([method_data[tile_fraction]['time_mean'] for tile_fraction in tile_fractions])
        time_stds = np.array([method_data[tile_fraction]['time_std'] for tile_fraction in tile_fractions])
        accuracy_means = np.array([method_data[tile_fraction]['accuracy_mean'] for tile_fraction in tile_fractions])
        accuracy_stds = np.array([method_data[tile_fraction]['accuracy_std'] for tile_fraction in tile_fractions])
        
        vectors[method_name] = {
            'tile_fractions': tile_fractions,
            'time_means': time_means,
            'time_stds': time_stds,
            'accuracy_means': accuracy_means,
            'accuracy_stds': accuracy_stds
        }
    
    return vectors


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def parse_data(path: Path):
    with open(path, 'r') as file:
        data = file.read()

        res = parse_data_to_dict(data)
        return convert_to_vectors(res)
    