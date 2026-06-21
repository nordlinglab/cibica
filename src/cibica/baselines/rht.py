"""
RHT: Randomized Hough Transform for Circle Detection

Implements a randomized version of the Hough transform for robust circle
fitting. Uses random sampling and voting to detect circles efficiently.

Main function: rht(edgels, **kwargs)
"""

import numpy as np
from scipy.special import comb


def rht(edgels, num_iterations=1000, threshold=5):
    """
    Fit a circle to edge points using Randomized Hough Transform.
    
    This algorithm repeatedly samples random triplets of points, fits circles
    to them, and counts how many points support each circle hypothesis.
    The circle with the most support (votes) is returned.
    
    Algorithm steps:
    1. Randomly sample 3 points
    2. Fit a circle through these 3 points
    3. Count how many points are within threshold distance of the circle
    4. Keep track of the best-scoring circle
    5. Repeat for num_iterations or until all combinations exhausted
    
    Parameters
    ----------
    edgels : numpy.ndarray
        Array of shape (n, 2) containing (x, y) coordinates of edge points
    num_iterations : int, optional
        Maximum number of random samples to try (default: 1000)
        If there are fewer than num_iterations possible triplet combinations,
        all combinations are considered
    threshold : float, optional
        Distance threshold for inlier counting (default: 5)
        Points within this distance from the circle are counted as votes
        
    Returns
    -------
    center : numpy.ndarray
        Array [cx, cy] of circle center coordinates
    radius : float
        Circle radius
        
    Notes
    -----
    Returns ([-1, -1], -1) if no valid circle is found.
    
    The algorithm automatically limits iterations to C(n,3) to avoid
    sampling the same triplet multiple times.
    
    Typical threshold values depend on image scale:
    - Small images (100x100): threshold ≈ 2-3 pixels
    - Medium images (512x512): threshold ≈ 5-10 pixels
    - Large images (1024x1024): threshold ≈ 10-20 pixels
    
    Performance scales as O(num_iterations * n) where n is number of points.
    
    Examples
    --------
    >>> theta = np.linspace(0, 2*np.pi, 100)
    >>> edgels = np.column_stack([50 + 20*np.cos(theta), 50 + 20*np.sin(theta)])
    >>> center, radius = rht(edgels, num_iterations=500, threshold=3)
    >>> print(f"Center: {center}, Radius: {radius:.2f}")
    """
    # Input validation
    if len(edgels) < 3:
        return np.array([-1, -1]), -1
    
    n_points = edgels.shape[0]
    
    # Limit iterations to number of possible triplet combinations
    max_combinations = int(comb(n_points, 3, exact=True))
    num_iterations = min(num_iterations, max_combinations)
    
    best_circle = None
    best_score = 0
    
    for iteration in range(num_iterations):
        # Randomly select 3 points
        indices = np.random.choice(n_points, 3, replace=False)
        sample = edgels[indices]
        
        # Construct linear system to solve for circle parameters
        # Circle equation: x² + y² + ax + by + c = 0
        # We solve: A * [a, b, c]^T = b
        A = np.column_stack([sample[:, 0], sample[:, 1], np.ones(3)])
        b = -np.sum(sample**2, axis=1)
        
        # Check if points are collinear (singular matrix)
        det = np.linalg.det(A)
        if np.abs(det) < 1e-10:
            continue  # Skip degenerate configuration
        
        try:
            # Solve for circle parameters
            x = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            continue  # Skip if solve fails
        
        # Convert to geometric parameters
        # Center: (-a/2, -b/2)
        # Radius: sqrt((a² + b²)/4 - c)
        center_x = -x[0] / 2
        center_y = -x[1] / 2
        radius_squared = (x[0]**2 + x[1]**2) / 4 - x[2]
        
        # Skip if radius is invalid (negative or too small)
        if radius_squared <= 0:
            continue
        
        radius = np.sqrt(radius_squared)
        
        # Count inliers: points within threshold distance of circle
        distances_to_center = np.sqrt((edgels[:, 0] - center_x)**2 + 
                                      (edgels[:, 1] - center_y)**2)
        geometric_distances = np.abs(distances_to_center - radius)
        score = np.sum(geometric_distances < threshold)
        
        # Update best circle if this one has more support
        if score > best_score:
            best_score = score
            best_circle = (center_x, center_y, radius)
    
    # Return best circle found
    if best_circle is None:
        return np.array([-1, -1]), -1
    
    center = np.array([best_circle[0], best_circle[1]])
    radius = best_circle[2]
    
    return center, float(radius)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2 or sys.argv[1] == 'test':
        # Run built-in tests
        print("Testing RHT (Randomized Hough Transform)")
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
        
        center, radius = rht(edgels, num_iterations=500, threshold=2)
        print(f"True: center={true_center}, radius={true_radius}")
        print(f"Detected: center=({center[0]:.2f}, {center[1]:.2f}), radius={radius:.2f}")
        error = np.sqrt((center[0]-true_center[0])**2 + (center[1]-true_center[1])**2)
        print(f"Center error: {error:.4f}, Radius error: {abs(radius-true_radius):.4f}")
        
        # Test 2: Noisy circle
        print("\nTest 2: Noisy circle")
        noise = np.random.randn(100, 2) * 1.5
        edgels_noisy = edgels + noise
        
        center, radius = rht(edgels_noisy, num_iterations=1000, threshold=5)
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
        
        center, radius = rht(edgels_outliers, num_iterations=1000, threshold=5)
        print(f"Detected: center=({center[0]:.2f}, {center[1]:.2f}), radius={radius:.2f}")
        error = np.sqrt((center[0]-true_center[0])**2 + (center[1]-true_center[1])**2)
        print(f"Center error: {error:.4f}, Radius error: {abs(radius-true_radius):.4f}")
        
        # Test 4: Small example
        print("\nTest 4: Small dataset")
        small_edgels = np.array([
            [2, 0], [4, 0], [6, 2], [6, 4],
            [4, 6], [2, 6], [0, 4], [0, 2]
        ], dtype=float)
        
        center, radius = rht(small_edgels, num_iterations=100, threshold=1)
        print(f"Detected: center=({center[0]:.4f}, {center[1]:.4f}), radius={radius:.4f}")
        print(f"Expected: center≈(3, 3), radius≈3.1623")
        
        # Test 5: Partial arc
        print("\nTest 5: Partial arc (90 degrees)")
        theta_arc = np.linspace(0, np.pi/2, 50)
        edgels_arc = np.column_stack([
            true_center[0] + true_radius * np.cos(theta_arc),
            true_center[1] + true_radius * np.sin(theta_arc)
        ])
        
        center, radius = rht(edgels_arc, num_iterations=500, threshold=3)
        print(f"Detected: center=({center[0]:.2f}, {center[1]:.2f}), radius={radius:.2f}")
        error = np.sqrt((center[0]-true_center[0])**2 + (center[1]-true_center[1])**2)
        print(f"Center error: {error:.4f}, Radius error: {abs(radius-true_radius):.4f}")
        
        # Test 6: Effect of iterations
        print("\nTest 6: Testing iteration count effect")
        print("  With 100 iterations:")
        center, radius = rht(edgels_noisy, num_iterations=100, threshold=5)
        error = np.sqrt((center[0]-true_center[0])**2 + (center[1]-true_center[1])**2)
        print(f"    Center error: {error:.4f}")
        
        print("  With 500 iterations:")
        center, radius = rht(edgels_noisy, num_iterations=500, threshold=5)
        error = np.sqrt((center[0]-true_center[0])**2 + (center[1]-true_center[1])**2)
        print(f"    Center error: {error:.4f}")
        
        print("  With 2000 iterations:")
        center, radius = rht(edgels_noisy, num_iterations=2000, threshold=5)
        error = np.sqrt((center[0]-true_center[0])**2 + (center[1]-true_center[1])**2)
        print(f"    Center error: {error:.4f}")
        
        print("\n" + "=" * 60)
        print("Testing complete!")
    else:
        # Load edgels from CSV file
        edgels = np.loadtxt(sys.argv[1], delimiter=',')
        center, radius = rht(edgels)
        print(f"center=({center[0]:.2f}, {center[1]:.2f}), radius={radius:.2f}")