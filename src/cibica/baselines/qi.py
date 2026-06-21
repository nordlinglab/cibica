"""
Qi 2024: Iteratively Reweighted Least Squares (IRLS) Circle Fitting

Implements a robust circle fitting algorithm using iteratively reweighted
least squares with Huber-type weighting. The method combines algebraic
circle fitting with robust M-estimation.

Reference: Qi et al. (2024)

Main function: qi_2024(edgels, **kwargs)
"""

import numpy as np
from scipy import linalg


def _create_design_matrix(points):
    """
    Create design matrix X for algebraic circle fitting.
    
    The design matrix maps points to the coefficients of the implicit
    circle equation: A(x² + y²) + Bx + Cy + D = 0
    
    Parameters
    ----------
    points : numpy.ndarray
        Array of shape (n, 2) containing (x, y) coordinates
        
    Returns
    -------
    X : numpy.ndarray
        Design matrix of shape (n, 4) with columns [x²+y², x, y, 1]
    """
    x, y = points[:, 0], points[:, 1]
    z = x**2 + y**2
    return np.column_stack((z, x, y, np.ones(len(x))))


def _create_constraint_matrix(X):
    """
    Create constraint matrix N_H for hyperaccurate fitting.
    
    This matrix enforces geometric constraints that improve accuracy
    over standard algebraic methods like Pratt or Taubin.
    
    Parameters
    ----------
    X : numpy.ndarray
        Design matrix of shape (n, 4)
        
    Returns
    -------
    N_H : numpy.ndarray
        Constraint matrix of shape (4, 4)
    """
    z, x, y = X[:, 0], X[:, 1], X[:, 2]
    z_mean, x_mean, y_mean = np.mean(z), np.mean(x), np.mean(y)
    
    # Pratt constraint matrix
    N_P = np.array([[0 , 0, 0, -2], 
                    [0 , 1, 0,  0], 
                    [0 , 0, 1,  0], 
                    [-2, 0, 0,  0]])
    
    # Taubin constraint matrix (centered)
    N_T = np.array([
        [4*z_mean, 2*x_mean, 2*y_mean, 0],
        [2*x_mean, 1       , 0       , 0],
        [2*y_mean, 0       , 1       , 0],
        [0       , 0       , 0       , 0]
    ])
    
    # Hyperaccurate constraint (combination of Pratt and Taubin)
    N_H = 2 * N_T - N_P
    
    return N_H


def _update_weights(residuals, k0=1.5, k1=2.5):
    """
    Update weights using Huber-type robust weighting function.
    
    Weight function:
    - w = 1 for small residuals (|v| < k0)
    - w = smooth decay for moderate residuals (k0 ≤ |v| ≤ k1)
    - w = 0 for large residuals (|v| > k1) - complete outlier rejection
    
    Parameters
    ----------
    residuals : numpy.ndarray
        Algebraic residuals from current fit
    k0 : float, optional
        Inner threshold for downweighting (default: 1.5)
    k1 : float, optional
        Outer threshold for outlier rejection (default: 2.5)
        
    Returns
    -------
    weights : numpy.ndarray
        Robust weights in range [0, 1]
    """
    # Robust scale estimate using MAD
    sigma0 = 1.4826 * np.median(np.abs(residuals))
    
    # Standardized residuals
    v = residuals / sigma0
    
    # Initialize all weights to 1
    weights = np.ones_like(v)
    
    # Apply Huber-type weighting
    for i, vi in enumerate(v):
        avi = np.abs(vi)
        
        # Moderate residuals: smooth downweighting
        if k0 <= avi <= k1:
            weights[i] = (k0 / avi) * ((k1 - k0) / (k1 - avi))**-2
        
        # Large residuals: complete rejection
        elif avi > k1:
            weights[i] = 0
    
    return weights


def qi_2024(edgels, epsilon=1e-6, max_iterations=500, k0=1.5, k1=2.5):
    """
    Fit a circle to edge points using Qi 2024 IRLS method.
    
    This algorithm combines hyperaccurate algebraic circle fitting with
    iteratively reweighted least squares (IRLS) for robustness.
    
    Algorithm steps:
    1. Initialize with uniform weights
    2. Solve constrained eigenvalue problem for circle parameters
    3. Compute residuals and update weights using Huber function
    4. Repeat until convergence
    
    Parameters
    ----------
    edgels : numpy.ndarray
        Array of shape (n, 2) containing (x, y) coordinates of edge points
    epsilon : float, optional
        Convergence tolerance for parameter changes (default: 1e-6)
    max_iterations : int, optional
        Maximum number of IRLS iterations (default: 500)
    k0 : float, optional
        Inner threshold for robust weighting (default: 1.5)
        Points with |residual| < k0*sigma get full weight
    k1 : float, optional
        Outer threshold for outlier rejection (default: 2.5)
        Points with |residual| > k1*sigma are completely rejected
        
    Returns
    -------
    center : numpy.ndarray
        Array [cx, cy] of circle center coordinates
    radius : float
        Circle radius
        
    Notes
    -----
    The hyperaccurate constraint combines the best properties of Pratt
    and Taubin methods, providing better accuracy than either alone.
    
    The IRLS weighting provides robustness to outliers without requiring
    explicit outlier detection.
    
    Tuning parameters k0 and k1:
    - Smaller k0: More aggressive downweighting of deviations
    - Larger k1: More tolerant of outliers
    - Typical range: k0 ∈ [1.0, 2.0], k1 ∈ [2.0, 3.5]
    
    Examples
    --------
    >>> theta = np.linspace(0, 2*np.pi, 100)
    >>> edgels = np.column_stack([50 + 20*np.cos(theta), 50 + 20*np.sin(theta)])
    >>> center, radius = qi_2024(edgels)
    >>> print(f"Center: {center}, Radius: {radius:.2f}")
    """
    # Input validation
    if len(edgels) < 3:
        return np.array([-1, -1]), -1
    
    # Create design matrix
    X = _create_design_matrix(edgels)
    
    # Create constraint matrix
    N_H = _create_constraint_matrix(X)
    
    # Initialize weights (identity matrix = uniform weights)
    W = np.eye(len(edgels))
    
    # Initialize parameters
    xi_prev = np.zeros(4)
    
    # IRLS iterations
    for iteration in range(max_iterations):
        # Solve constrained generalized eigenvalue problem
        # M * xi = lambda * xi, where M = N_H^(-1) * X^T * W * X
        M = linalg.solve(N_H, X.T @ W @ X)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = linalg.eig(M)
        
        # Select eigenvector corresponding to smallest positive eigenvalue
        positive_indices = [i for i, val in enumerate(eigenvalues.real) if val > 0]
        
        if len(positive_indices) == 0:
            raise ValueError("No positive eigenvalues found - fitting failed")
        
        # Find index of smallest positive eigenvalue
        min_idx = positive_indices[np.argmin([eigenvalues.real[i] for i in positive_indices])]
        xi = eigenvectors[:, min_idx].real
        
        # Normalize by setting first coefficient (A) to 1
        xi = xi / xi[0]
        
        # Calculate algebraic residuals
        residuals = X @ xi
        
        # Update weights using robust Huber function
        W = np.diag(_update_weights(residuals, k0=k0, k1=k1))
        
        # Check convergence
        error = np.linalg.norm(xi - xi_prev)
        if error < epsilon:
            break
        
        xi_prev = xi
    
    # Convert algebraic parameters to geometric parameters
    # Circle equation: A(x² + y²) + Bx + Cy + D = 0
    # Center: (x0, y0) = (-B/2A, -C/2A)
    # Radius: r = sqrt((B² + C² - 4AD) / 4A²)
    A, B, C, D = xi
    
    x0 = -B / (2 * A)
    y0 = -C / (2 * A)
    r = np.sqrt((B**2 + C**2 - 4*A*D) / (4 * A**2))
    
    center = np.array([x0, y0])
    
    return center, float(r)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2 or sys.argv[1] == 'test':
        # Run built-in tests
        print("Testing Qi 2024 (IRLS Circle Fitting)")
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
        
        center, radius = qi_2024(edgels)
        print(f"True: center={true_center}, radius={true_radius}")
        print(f"Detected: center=({center[0]:.2f}, {center[1]:.2f}), radius={radius:.2f}")
        error = np.sqrt((center[0]-true_center[0])**2 + (center[1]-true_center[1])**2)
        print(f"Center error: {error:.6f}, Radius error: {abs(radius-true_radius):.6f}")
        
        # Test 2: Noisy circle
        print("\nTest 2: Noisy circle")
        noise = np.random.randn(100, 2) * 1.5
        edgels_noisy = edgels + noise
        
        center, radius = qi_2024(edgels_noisy)
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
        
        center, radius = qi_2024(edgels_outliers)
        print(f"Detected: center=({center[0]:.2f}, {center[1]:.2f}), radius={radius:.2f}")
        error = np.sqrt((center[0]-true_center[0])**2 + (center[1]-true_center[1])**2)
        print(f"Center error: {error:.4f}, Radius error: {abs(radius-true_radius):.4f}")
        
        # Test 4: Small example
        print("\nTest 4: Small dataset")
        small_edgels = np.array([
            [2, 0], [4, 0], [6, 2], [6, 4],
            [4, 6], [2, 6], [0, 4], [0, 2]
        ], dtype=float)
        
        center, radius = qi_2024(small_edgels)
        print(f"Detected: center=({center[0]:.4f}, {center[1]:.4f}), radius={radius:.4f}")
        print(f"Expected: center≈(3, 3), radius≈3.1623")
        
        # Test 5: Stress test with different k0, k1 values
        print("\nTest 5: Testing different robustness parameters")
        theta_circle = np.linspace(0, 2*np.pi, 70)
        circle_points = np.column_stack([
            true_center[0] + true_radius * np.cos(theta_circle),
            true_center[1] + true_radius * np.sin(theta_circle)
        ])
        outliers = np.random.rand(30, 2) * 100
        edgels_test = np.vstack([circle_points, outliers])
        
        print("  With k0=1.5, k1=2.5 (default):")
        center, radius = qi_2024(edgels_test, k0=1.5, k1=2.5)
        error = np.sqrt((center[0]-true_center[0])**2 + (center[1]-true_center[1])**2)
        print(f"    Center error: {error:.4f}, Radius error: {abs(radius-true_radius):.4f}")
        
        print("  With k0=1.0, k1=3.0 (more aggressive):")
        center, radius = qi_2024(edgels_test, k0=1.0, k1=3.0)
        error = np.sqrt((center[0]-true_center[0])**2 + (center[1]-true_center[1])**2)
        print(f"    Center error: {error:.4f}, Radius error: {abs(radius-true_radius):.4f}")
        
        print("\n" + "=" * 60)
        print("Testing complete!")
    else:
        # Load edgels from CSV file
        edgels = np.loadtxt(sys.argv[1], delimiter=',')
        center, radius = qi_2024(edgels)
        print(f"center=({center[0]:.2f}, {center[1]:.2f}), radius={radius:.2f}")