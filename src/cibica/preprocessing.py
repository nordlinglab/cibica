import cv2
import numpy as np
import os
from pathlib import Path

DATA_DIR = Path(__file__).parent / 'data'

def auto_canny(image, sigma=0.33):
    """
    Auto threshold Canny edge detector.
    
    Automatically determines thresholds based on image median.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input image (single channel)
    sigma : float
        Threshold factor (default: 0.33)
        
    Returns
    -------
    edged : numpy.ndarray
        Binary edge image
        
    Reference
    ---------
    https://github.com/oleg-Shipitko/Image-processing-Python/blob/master/auto_canny.py
    """
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged

def frames_to_edgepoints(img):
    """
    Extract edge point coordinates from binary edge image.
    
    Parameters
    ----------
    img : numpy.ndarray
        Binary edge image (0-1 or 0-255)
        
    Returns
    -------
    XY : numpy.ndarray
        Edge point coordinates, shape (N, 2) as [[row, col], ...]
    """
    img = img / np.max(img)
    x, y = np.where(img == 1)
    XY = np.zeros((len(x), 2))
    for i in range(len(x)):
        XY[i, 0] = x[i]
        XY[i, 1] = y[i]
    return XY

def preprocess_green_level(BS_crop, green_level):
    """
    Preprocess image using green level thresholding in HSV space.
    
    Parameters
    ----------
    BS_crop : numpy.ndarray
        Input BGR image
    green_level : int
        Green threshold level (70-86 recommended)
        
    Returns
    -------
    GreenMask : numpy.ndarray
        Binary mask after thresholding
    GreenCanny : numpy.ndarray
        Edge image (0-255 values)
    edgels : numpy.ndarray
        Edge point coordinates, shape (N, 2) as [[row, col], ...]
    """
    BS_HSV = cv2.cvtColor(BS_crop, cv2.COLOR_BGR2HSV)
    greenlower = np.array([36, green_level, green_level], dtype=np.uint8)
    greenupper = np.array([86, 255, 255], dtype=np.uint8)
    GreenMask = cv2.inRange(BS_HSV, greenlower, greenupper)
    GreenCanny = auto_canny(np.uint8(GreenMask))
    edgels = frames_to_edgepoints(GreenCanny / 255)
    return GreenMask, GreenCanny, edgels

def preprocess_median_filter(BS_crop, G_crop, median_size):
    """
    Preprocess image using median filtering.
    
    Parameters
    ----------
    BS_crop : numpy.ndarray
        Input BGR image (black sphere region)
    G_crop : numpy.ndarray
        Green background reference image
    median_size : int
        Median filter kernel size (must be odd: 3, 5, 7, ...)
        
    Returns
    -------
    GreenMask : numpy.ndarray
        Binary mask after filtering and thresholding
    GreenCanny : numpy.ndarray
        Edge image (0-255 values)
    edgels : numpy.ndarray
        Edge point coordinates, shape (N, 2) as [[row, col], ...]
    """
    # Calculate green color range from reference
    rows, cols = G_crop.shape[0], G_crop.shape[1]
    green = np.zeros((rows * cols, 3))
    G_crop_HSV = cv2.cvtColor(G_crop, cv2.COLOR_BGR2HSV)
    
    for ch in range(3):
        for row in range(rows):
            for col in range(cols):
                green[row * cols + col, ch] = G_crop_HSV[row, col, ch]
    
    green_std = np.std(green, axis=0)
    green_min = np.min(green, axis=0)
    green_max = np.max(green, axis=0)
    GL = np.maximum(0, np.floor(green_min - 5 * green_std))
    GU = np.minimum(255, np.ceil(green_max + 5 * green_std))
    
    # Apply median filter
    median_filter = cv2.medianBlur(BS_crop, median_size)
    HSV = cv2.cvtColor(median_filter, cv2.COLOR_BGR2HSV)
    GreenMask = cv2.inRange(HSV, GL, GU)
    GreenCanny = auto_canny(np.uint8(GreenMask))
    edgels = frames_to_edgepoints(GreenCanny / 255)
    
    return GreenMask, GreenCanny, edgels

def get_preprocessing_configs():
    """
    Original preprocessing configuration generator.

    Returns
    -------
    configs : list of dict
        Each dict contains:
            - 'name'         : str  (e.g. 'GL70', 'Med3')
            - 'green_level'  : int or None
            - 'median_size'  : int or None
    """
    configs = []

    # Green-level thresholding
    for green_level in [70, 72, 74, 76, 78, 80, 82, 84, 86]:
        configs.append({
            "name": f"GL{green_level}",
            "green_level": green_level,
            "median_size": None
        })

    # Median filtering
    for median_size in [3, 5, 7, 9, 11, 13, 15, 17, 19]:
        configs.append({
            "name": f"Med{median_size}",
            "green_level": None,
            "median_size": median_size
        })

    return configs


def preprocess_image(filename, method='green_level', param=76, hough=False):
    """
    Load and preprocess image from ROI folders.
    
    Parameters
    ----------
    filename : str
        Image filename (without .png extension)
    method : str
        'green_level' or 'median_filter'
    param : int
        For green_level: threshold value (70-86)
        For median_filter: kernel size (3, 5, 7, ...)
    hough : bool
        If True, returns edge image for HOUGH (default: False)
        If False, returns edgels for CIBICA
        
    Returns
    -------
    data : numpy.ndarray
        If hough=True: binary edge image (0-1 values)
        If hough=False: edge point coordinates, shape (N, 2)
    GreenMask : numpy.ndarray
        Binary mask
    """
    # Load images
    bs_path = DATA_DIR / 'black_sphere_ROI' / (filename + '.png')
    gb_path = DATA_DIR / 'green_back_ROI' / (filename + '.png')
    
    BS_crop = cv2.imread(str(bs_path))
    G_crop = cv2.imread(str(gb_path))
    
    if BS_crop is None:
        raise FileNotFoundError(f"Image not found: {bs_path}")
    
    # Preprocess based on method
    if method == 'green_level':
        GreenMask, GreenCanny, edgels = preprocess_green_level(BS_crop, param)
    elif method == 'median_filter':
        GreenMask, GreenCanny, edgels = preprocess_median_filter(BS_crop, G_crop, param)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Return appropriate format
    if hough:
        return GreenMask  # HOUGH uses this directly
    else:
        return edgels  # CIBICA uses this


if __name__ == "__main__":
    """Test preprocessing functions"""
    print("Preprocessing configurations:")
    configs = get_preprocessing_configs()
    for i, config in enumerate(configs):
        print(f"{i+1:2d}. {config['name']}")
    print(f"\nTotal: {len(configs)} configurations")
