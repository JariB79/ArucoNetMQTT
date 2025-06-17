import numpy as np


def get_camera_matrix():
    c, Lx, Ly = -1.358, 0.0022, 0.0022
    fx, fy, cx, cy = c/Lx, -c/Ly, 324.6594, 245.4463
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    return camera_matrix

def get_dist_coeffs():
    k1, k2, p1, p2, k3 = -0.0154, 0.1551, 0, 0, 0
    dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)
    return dist_coeffs

