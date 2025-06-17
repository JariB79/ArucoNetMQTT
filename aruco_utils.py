import numpy as np
import cv2


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
    # rvecs als Rotationsmatrix in Form [rx, ry, rz]
    # tves als Translationsmatrix in Form [tx, ty, tz]
    success, rvecs, tvecs = cv2.solvePnP(marker_3d_points, corners[0], camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE)
    print("rvecs: ", rvecs)
    print("tvecs: ", tvecs)

    if success:
        return np.array(rvecs, dtype=np.float32), np.array(tvecs, dtype=np.float32)
    else:
        return None, None


def convert_marker_to_cube(ids, tvecs, rvecs):
    """
    Converts an ArUco marker into a cube representation with the correct reference point.

    Args:
        ids (int): ArUco marker ID used to determine the cube and face.
        tvecs (numpy.ndarray): Translation vector representing the marker's position.
        rvecs (numpy.ndarray): Rotation vector representing the marker's orientation.

    Returns:
        dict: Cube information including ID, face, position (x, y, z), and corrected rotation angle.
    """
    cube_id = ids // 10  # Determines the cube (team number)
    face_id = ids % 10   # Determines the side of the cube

    # Offset of the reference point (1.5 cm behind marker)
    offset = np.array([0, 0, 0.015], dtype=np.float32)
    cube_position = tvecs.flatten() + offset

    # Position on the cube (front, back, etc.)
    face_positions = {
        0: "Vorne",
        1: "Rechts",
        2: "Hinten",
        3: "Links",
        4: "Oben",
        5: "Unten"
    }

    # Yaw correction based on the recognized side
    face_yaw_correction = {
        0: 0,  # Front → 0°
        1: 90,  # Right → 90°
        2: 180,  # Back → 180°
        3: -90,  # Left → -90°
    }

    # Original rotation of the ArUco marker
    raw_yaw = float(rvecs.flatten()[2] * (180 / np.pi))

    # Final yaw angle of the cube, taking into account the camera view
    adjusted_yaw = raw_yaw + face_yaw_correction.get(face_id, 0)

    return {
        "cube_id": int(cube_id),
        "face": face_positions.get(face_id, "Unbekannt"),
        "Position": {
            "x": float(cube_position[0]),
            "y": float(cube_position[1]),
            "z": float(cube_position[2])
        },
        "Rotation": adjusted_yaw  # Corrected yaw angle
    }