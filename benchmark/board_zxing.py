import numpy as np
import cv2
import zxingcpp
from board_reliability_benchmark import full_benchmark

def main():
    iterations = 100
    # print("Testing simultaneous method")
    # full_benchmark(iterations, detect_simultaneous)

    #print("Testing vectorised method")
    #full_benchmark(iterations, detect_vectorise)

    print("Testing iterative method")
    full_benchmark(iterations, detect_iterative)

def detect_iterative(board_img):
    def segment(board_img):
        squares = []
        rows = np.vsplit(board_img, 8)
        for row in rows:
            squares.append(np.hsplit(row, 8))
        
        return squares
    
    squares = segment(board_img)
    board = np.full((8, 8), -1, dtype=np.int16)

    for i, row in enumerate(squares):
        for j, square in enumerate(row):
            barcodes = zxingcpp.read_barcodes(square, binarizer=zxingcpp.Binarizer.GlobalHistogram)
            if len(barcodes) == 1: # Tile on square
                board[i, j] = barcodes[0].text

    return board

def detect_simultaneous(board_img):
    def pos_to_point(pos):
        return (pos.x, pos.y)

    n = board_img.shape[0]
    subarray_size = n // 8
    def get_index(bounding_box):
        # assert len(bounding_box) == 4 we need 4 points
        y, x = bounding_box.mean(axis=0) // subarray_size
        return int(x), int(y)

    board = np.full((8, 8), -1, dtype=np.int16)
    barcodes = zxingcpp.read_barcodes(board_img, binarizer=zxingcpp.Binarizer.GlobalHistogram)
    for code in barcodes:
        bb = [code.position.top_left, code.position.top_right, code.position.bottom_left, code.position.bottom_right]
        idx = get_index(np.array([pos_to_point(p) for p in bb]))
        assert board[idx] == -1
        board[idx] = code.text
    
    return board

if __name__ == '__main__':
    main()