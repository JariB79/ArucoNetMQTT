import numpy as np
import cv2
import cv2.aruco as aruco


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


def get_aruco_markers(frame, camera_matrix, MARKER_SIZE, dist_coeffs):
    """
    Detects ArUco markers in the given frame.
    Returns a list of detected markers with their IDs, distances, and angles.
    """
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(frame)

    if ids is not None:
        ids = ids.flatten()
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, camera_matrix,
                                                          dist_coeffs)
        rvecs = rvecs[0].tolist()
        tvecs = tvecs[0].tolist()
        return ids, rvecs, tvecs
    else:
        return [], [], []

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


def convert_marker_to_cube(ids, tvecs, rvecs):
    """
    Converts an ArUco marker into a cube representation with the correct reference point.
    Computes the global viewing direction of the camera and the distance between camera and cube.

    Args:
        ids (int): ArUco marker ID used to determine the cube and face.
        tvecs (numpy.ndarray): Translation vector representing the marker's position.
        rvecs (numpy.ndarray): Rotation vector representing the marker's orientation.

    Returns:
        dict: Cube information including ID, face, position, distance, and global viewing angle.
    """
    cube_id = ids // 10
    face_id = ids % 10

    # Offset from marker center (still raw, as besprochen)
    offset = np.array([0, 0, 0.015], dtype=np.float32)
    cube_position = tvecs.flatten() + offset

    # Zuordnung der Seiten
    face_positions = {
        0: "Vorne",
        1: "Rechts",
        2: "Hinten",
        3: "Links",
        4: "Oben",
        5: "Unten"
    }

    # Welche Seite zeigt wohin? (also wo steht die Kamera, wenn sie frontal draufblickt)
    marker_face_to_global_view = {
        0: 180,  # Kamera schaut frontal auf "Vorne", steht also auf +x → Blickrichtung -x = 180°
        1: 0,    # Kamera steht auf +y → Blickrichtung -y = 0°
        2: 270,  # Kamera steht auf -x → Blickrichtung +x = 270°
        3: 90    # Kamera steht auf -y → Blickrichtung +y = 90°
    }

    # Roher Rotationswert (rvecs[0]) in Grad
    yaw_local = float(rvecs.flatten()[0] * (180 / np.pi))

    # Globale Blickrichtung der Kamera
    if face_id in marker_face_to_global_view:
        camera_angle_global = (marker_face_to_global_view[face_id] + yaw_local) % 360
    else:
        camera_angle_global = None  # fallback für ungültige Marker

    # Distanz zwischen Kamera und Marker (in cm, da MQTT cm erwartet)
    distance_cm = float(np.linalg.norm(tvecs.flatten()) * 100)

    return {
        "cube_id": int(cube_id),
        "face": face_positions.get(face_id, "Unbekannt"),
        "Position": {
            "x": float(cube_position[0]),
            "y": float(cube_position[1]),
            "z": float(cube_position[2])
        },
        "Distance_cm": round(distance_cm, 2),
        "View_angle_deg": round(camera_angle_global, 1) if camera_angle_global is not None else "N/A"
    }

def calc_polar_coordinates(cube_data):
    # Calculate  polar coordinates
    x, y = cube_data["Position"]["z"], cube_data["Position"]["x"]
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x) * (180 / np.pi)

    # Yaw angle
    yaw = cube_data["Rotation_deg"]

    payload = {
        "id": 4,
        "Others": [
            {
                "id": cube_data["cube_id"],
                "face": cube_data["face"],
                "Position": {
                    "Distance": float(round(r, 3)),  # Abstand
                    "Angle_deg": float(round(theta, 2)),  # Richtung
                    "Yaw_deg": float(round(yaw, 2))  # Rotation
                    # "x": float(cube_data["Position"]["x"]),
                    # "y": float(cube_data["Position"]["y"]),
                    # "z": float(cube_data["Position"]["z"])
                }
            }
        ]
    }
    return payload


