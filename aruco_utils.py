import numpy as np
import cv2
import cv2.aruco as aruco


def get_aruco_detector():
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    return detector

# 3D coordinates of the marker corner points (for flat marker on the XY plane)
def get_marker_3d_points(MARKER_SIZE):
    """
    Returns the 3D coordinates of an ArUco marker's corners.
    The marker is assumed to be flat on the XY plane, with its center at the origin.

    Args:
        MARKER_SIZE (float): Physical size of the marker in meters.

    Returns:
        numpy.ndarray: 3D corner points in a NumPy array.
    """
    half_size = MARKER_SIZE / 2
    return np.array([
        [-half_size, half_size, 0],
        [half_size, half_size, 0],
        [half_size, -half_size, 0],
        [-half_size, -half_size, 0]
    ], dtype=np.float32)

def estimate_pose(corners, MARKER_SIZE, camera_matrix, dist_coeffs):
    """
       Estimates the pose of an ArUco marker using solvePnP.

       Args:
           corners (numpy.ndarray): Detected marker corners.
           MARKER_SIZE (float): Physical marker size in meters.
           camera_matrix (numpy.ndarray): Intrinsic camera matrix.
           dist_coeffs (numpy.ndarray): Camera distortion coefficients.

       Returns:
           tuple: (rvecs, tvecs) rotation and translation vectors or (None, None) on failure.
       """
    marker_3d_points = get_marker_3d_points(MARKER_SIZE)
    success, rvecs, tvecs = cv2.solvePnP(marker_3d_points, corners[0], camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE)

    if success:
        return np.array(rvecs, dtype=np.float32), np.array(tvecs, dtype=np.float32)
    else:
        return None, None


