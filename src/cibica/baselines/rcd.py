"""
RCD: RANSAC Circle Detection with Distance Constraints

Implements a RANSAC-based circle fitting algorithm with additional constraints
on point separation. Samples 4 points and tries all possible triplet combinations,
requiring minimum separation between sampled points.

Main function: rcd(edgels, **kwargs)
"""

import numpy as np


def _distance(p1, p2):
    """
    Calculate Euclidean distance between two points.
    
    Parameters
    ----------
    p1, p2 : array-like
        Points of shape (2,) containing (x, y) coordinates
        
    Returns
    -------
    float
        Euclidean distance between the points
    """
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def _circle_from_points(p1, p2, p3):
    """
    Calculate circle parameters from three points.
    
    Uses the determinant method to find the unique circle passing through
    three non-collinear points.
    
    Parameters
    ----------
    p1, p2, p3 : array-like
        Points of shape (2,) containing (x, y) coordinates
        
    Returns
    -------
    center : numpy.ndarray or None
        Circle center (cx, cy), or None if points are collinear
    radius : float or None
        Circle radius, or None if points are collinear
    """
    temp = p2[0]**2 + p2[1]**2
    bc = (p1[0]**2 + p1[1]**2 - temp) / 2
    cd = (temp - p3[0]**2 - p3[1]**2) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])
    
    # Check if points are collinear
    if abs(det) < 1.0e-6:
        return None, None
    
    # Calculate center
    cx = (bc * (p2[1] - p3[1]) - cd * (p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det
    center = np.array([cx, cy])
    
    # Calculate radius as mean distance from center to the three points
    radius = np.mean([_distance(center, p1), 
                     _distance(center, p2), 
                     _distance(center, p3)])
    
    return center, radius


def rcd(edgels, num_iterations=1000, distance_threshold=2, 
        min_inliers=10, min_distance=20):
    """
    Fit a circle to edge points using RANSAC with Distance Constraints.
    
    This algorithm extends RANSAC by:
    1. Sampling 4 points instead of 3
    2. Trying all 4 possible triplet combinations from the 4 points
    3. Requiring minimum separation between sampled points
    4. Selecting the circle with most inliers above a minimum threshold
    
    The additional constraints help avoid degenerate configurations and
    improve robustness in cluttered environments.
    
    Algorithm steps:
    1. Randomly sample 4 points
    2. For each of 4 possible triplets from these points:
       a. Check if points are sufficiently separated
       b. Fit circle to the triplet
       c. Count inliers within distance_threshold
    3. Keep best circle with most inliers (if above min_inliers)
    4. Repeat for num_iterations
    
    Parameters
    ----------
    edgels : numpy.ndarray
        Array of shape (n, 2) containing (x, y) coordinates of edge points
    num_iterations : int, optional
        Number of RANSAC iterations (default: 1000)
    distance_threshold : float, optional
        Maximum distance for a point to be considered an inlier (default: 2)
    min_inliers : int, optional
        Minimum number of inliers required for a valid circle (default: 10)
    min_distance : float, optional
        Minimum required separation between sampled points (default: 20)
        This prevents degenerate configurations from clustered points
        
    Returns
    -------
    center : numpy.ndarray
        Array [cx, cy] of circle center coordinates
    radius : float
        Circle radius
        
    Notes
    -----
    Returns ([-1, -1], -1) if no valid circle is found.
    
    The algorithm tries 4*num_iterations triplet fits (4 triplets per iteration),
    making it more thorough than standard RANSAC but also ~4x slower.
    
    The min_distance constraint is particularly useful when:
    - Points are clustered in groups
    - Image has multiple overlapping structures
    - Noise creates tight clusters of false edges
    
    Recommended parameter ranges:
    - distance_threshold: 1-5 pixels (depends on noise level)
    - min_inliers: 5-20 (depends on expected arc length)
    - min_distance: 10-50 pixels (depends on expected radius)
    
    Examples
    --------
    >>> theta = np.linspace(0, 2*np.pi, 100)
    >>> edgels = np.column_stack([50 + 20*np.cos(theta), 50 + 20*np.sin(theta)])
    >>> center, radius = rcd(edgels, num_iterations=500, distance_threshold=3)
    >>> print(f"Center: {center}, Radius: {radius:.2f}")
    """
    # Input validation
    if len(edgels) < 4:
        return np.array([-1, -1]), -1
    
    best_circle = None
    best_inliers = 0
    
    for iteration in range(num_iterations):
        # Randomly sample 4 points
        indices = np.random.choice(edgels.shape[0], 4, replace=False)
        sample = edgels[indices]
        
        # Try all 4 possible triplet combinations (leave one out)
        for i in range(4):
            # Select 3 points (excluding point i)
            triplet_indices = [j for j in range(4) if j != i]
            p1 = sample[triplet_indices[0]]
            p2 = sample[triplet_indices[1]]
            p3 = sample[triplet_indices[2]]
            
            # Check if points are sufficiently separated
            d12 = _distance(p1, p2)
            d23 = _distance(p2, p3)
            d31 = _distance(p3, p1)
            
            if (d12 > min_distance and 
                d23 > min_distance and 
                d31 > min_distance):
                
                # Fit circle to the triplet
                center, radius = _circle_from_points(p1, p2, p3)
                
                if center is not None:
                    # Calculate geometric distances from all points to circle
                    distances_to_center = np.sqrt(np.sum((edgels - center)**2, axis=1))
                    geometric_distances = np.abs(distances_to_center - radius)
                    
                    # Count inliers
                    inliers = np.sum(geometric_distances < distance_threshold)
                    
                    # Update best circle if this one is better
                    if inliers > best_inliers and inliers >= min_inliers:
                        best_inliers = inliers
                        best_circle = (center, radius)
    
    # Return best circle found
    if best_circle is None:
        return np.array([-1, -1]), -1
    
    return best_circle[0], float(best_circle[1])


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2 or sys.argv[1] == 'test':
        # Run built-in tests
        print("Testing RCD (RANSAC Circle Detection)")
        print("=" * 60)
        
        # Test 1: Perfect circle
        print("\nTest 1: Perfect circle (no outliers)")
        theta = np.linspace(0, 2*np.pi, 100)
        true_center = (50, 50)
        true_radius = 20
        edgels = np.column_stack([
            true_center[0] + true_radius * np.cos(theta),
            true_center[1] + true_radius * np.sin(theta)
        ])
        
        center, radius = rcd(edgels, num_iterations=500, distance_threshold=2, 
                            min_inliers=50, min_distance=10)
        print(f"True: center={true_center}, radius={true_radius}")
        print(f"Detected: center=({center[0]:.2f}, {center[1]:.2f}), radius={radius:.2f}")
        error = np.sqrt((center[0]-true_center[0])**2 + (center[1]-true_center[1])**2)
        print(f"Center error: {error:.4f}, Radius error: {abs(radius-true_radius):.4f}")
        
        # Test 2: Noisy circle
        print("\nTest 2: Noisy circle")
        noise = np.random.randn(100, 2) * 1.5
        edgels_noisy = edgels + noise
        
        center, radius = rcd(edgels_noisy, num_iterations=1000, distance_threshold=5,
                            min_inliers=40, min_distance=8)
        print(f"Detected: center=({center[0]:.2f}, {center[1]:.2f}), radius={radius:.2f}")
        error = np.sqrt((center[0]-true_center[0])**2 + (center[1]-true_center[1])**2)
        print(f"Center error: {error:.4f}, Radius error: {abs(radius-true_radius):.4f}")
        
        # Test 3: Circle with 20% outliers
        print("\nTest 3: Circle with 20% outliers")
        theta_circle = np.linspace(0, 2*np.pi, 80)
        circle_points = np.column_stack([
            true_center[0] + true_radius * np.cos(theta_circle),
            true_center[1] + true_radius * np.sin(theta_circle)
        ])
        outliers = np.random.rand(20, 2) * 100
        edgels_outliers = np.vstack([circle_points, outliers])
        
        center, radius = rcd(edgels_outliers, num_iterations=1000, distance_threshold=3,
                            min_inliers=30, min_distance=10)
        print(f"Detected: center=({center[0]:.2f}, {center[1]:.2f}), radius={radius:.2f}")
        error = np.sqrt((center[0]-true_center[0])**2 + (center[1]-true_center[1])**2)
        print(f"Center error: {error:.4f}, Radius error: {abs(radius-true_radius):.4f}")
        
        # Test 4: Partial arc
        print("\nTest 4: Partial arc (90 degrees)")
        theta_arc = np.linspace(0, np.pi/2, 50)
        edgels_arc = np.column_stack([
            true_center[0] + true_radius * np.cos(theta_arc),
            true_center[1] + true_radius * np.sin(theta_arc)
        ])
        
        center, radius = rcd(edgels_arc, num_iterations=500, distance_threshold=2,
                            min_inliers=20, min_distance=8)
        print(f"Detected: center=({center[0]:.2f}, {center[1]:.2f}), radius={radius:.2f}")
        error = np.sqrt((center[0]-true_center[0])**2 + (center[1]-true_center[1])**2)
        print(f"Center error: {error:.4f}, Radius error: {abs(radius-true_radius):.4f}")
        
        # Test 5: Effect of min_distance constraint
        print("\nTest 5: Testing min_distance constraint")
        # Add some clustered noise points
        clusters = np.random.randn(30, 2) * 3 + [25, 25]
        edgels_clustered = np.vstack([circle_points, clusters])
        
        print("  With min_distance=5 (loose constraint):")
        center, radius = rcd(edgels_clustered, num_iterations=500, 
                            distance_threshold=3, min_inliers=30, min_distance=5)
        error = np.sqrt((center[0]-true_center[0])**2 + (center[1]-true_center[1])**2)
        print(f"    Center error: {error:.4f}")
        
        print("  With min_distance=15 (strict constraint):")
        center, radius = rcd(edgels_clustered, num_iterations=500,
                            distance_threshold=3, min_inliers=30, min_distance=15)
        error = np.sqrt((center[0]-true_center[0])**2 + (center[1]-true_center[1])**2)
        print(f"    Center error: {error:.4f}")
        
        print("\n" + "=" * 60)
        print("Testing complete!")
    else:
        # Load edgels from CSV file
        edgels = np.loadtxt(sys.argv[1], delimiter=',')
        center, radius = rcd(edgels)
        print(f"center=({center[0]:.2f}, {center[1]:.2f}), radius={radius:.2f}")