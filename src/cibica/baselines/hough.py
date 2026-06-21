"""
Hough Transform Circle Detection

Reference:
    Duda, R. O., & Hart, P. E. (1972). Use of the Hough transformation to detect
    lines and curves in pictures. Communications of the ACM, 15(1), 11-15.
    
    OpenCV Implementation: Yuen, H. K., et al. (1990). Comparative study of 
    Hough transform methods for circle finding. Image and Vision Computing, 8(1), 71-77.
"""

import cv2
import numpy as np


def HOUGH(img, minDist=300, param1=50, param2=8, minRadius=5, maxRadius=25):
    """
    Detect circles using Hough Circle Transform.
    
    This is a wrapper around OpenCV's HoughCircles implementation with
    parameters tuned for the circle detection task.
    
    Parameters
    ----------
    img : numpy.ndarray
        Input image (grayscale, typically preprocessed/thresholded)
    minDist : int
        Minimum distance between detected circle centers (default: 300)
    param1 : int
        Higher threshold for Canny edge detector (default: 50)
    param2 : int
        Accumulator threshold for circle centers (default: 8)
        Lower values lead to more (possibly false) detections
    minRadius : int
        Minimum circle radius to detect (default: 5)
    maxRadius : int
        Maximum circle radius to detect (default: 25)
        
    Returns
    -------
    x, y, r : float
        Circle center (x, y) and radius r
        Returns (0, 0, 0) if no circle detected
        
    Examples
    --------
    >>> # Use with preprocessed image
    >>> from preprocessing import extract_edgels
    >>> _, binary = extract_edgels(image, method='green_level', threshold=76)
    >>> x, y, r = HOUGH(binary)
    
    Notes
    -----
    The parameters used here (param2=8, minDist=300) are tuned for the
    specific application in the paper. You may need to adjust these for
    different image sizes or circle sizes.
    """
    Hough = cv2.HoughCircles(
        img, 
        cv2.HOUGH_GRADIENT, 
        1, 
        minDist, 
        param1=param1, 
        param2=param2, 
        minRadius=minRadius, 
        maxRadius=maxRadius
    )
    
    if Hough is None:
        return 0, 0, 0
    else:
        Hough = np.uint16(np.around(Hough))
        HoughVal = Hough[0, 0, :]
        XC, YC, R = HoughVal[0], HoughVal[1], HoughVal[2]
        return float(XC), float(YC), float(R)
