import cv2
import cv2.aruco as aruco
import numpy as np
import requests
import paho.mqtt.client as mqtt
import json


# Marker-Größe in Metern
MARKER_SIZE = 0.025

# ESP32-CAM IP-Adresse
ip_address = "192.168.0.156"

# URL für das Setzen der Auflösung auf  VGA (640x480)
resolution_url = (f"http://{ip_address}/control?var=framesize&val=10")

# Anfrage senden, um die Auflösung zu ändern
response = requests.get(resolution_url)

if response.status_code == 200:
    print("Auflösung erfolgreich auf VGA (640x480) gesetzt!")
else:
    print("Fehler beim Setzen der Auflösung:", response.status_code)

url = f'http://{ip_address}:81/stream'

# Variablen für Kamerakalibrierung
c, Lx, Ly = -1.358, 0.0022, 0.0022
fx, fy, cx, cy = c/Lx, -c/Ly, 324.6594, 245.4463
k1, k2, p1, p2, k3 = -0.0154, 0.1551, 0, 0, 0

camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)


# MQTT Einstellungen
MQTT_BROKER = "192.168.0.252"
MQTT_PORT = 1883
MQTT_TOPIC_PUBLISH = "aruco/detection"
MQTT_TOPIC_SUBSCRIBE = "aruco/detection"

# 3D-Koordinaten der Marker-Eckpunkte (bei flachem Marker auf der XY-Ebene)
def get_marker_3d_points(MARKER_SIZE):
    half_size = MARKER_SIZE / 2
    return np.array([
        [-half_size, half_size, 0],
        [half_size, half_size, 0],
        [half_size, -half_size, 0],
        [-half_size, -half_size, 0]
    ], dtype=np.float32)


# Berechnung der Pose mit solvePnP
def estimate_pose(corners, MARKER_SIZE, camera_matrix, dist_coeffs):
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


# Callback-Funktion für empfangene MQTT-Nachrichten
def on_message(client, userdata, msg):
    print(f"Empfangene Nachricht: {msg.topic}: {msg.payload.decode()}")

# Initialisiere MQTT-Client
client = mqtt.Client()
client.on_message = on_message
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.subscribe(MQTT_TOPIC_SUBSCRIBE)
client.loop_start()


def convert_marker_to_cube(ids, tvecs, rvecs):
    """
    Konvertiert einen ArUco-Marker in einen Würfel mit korrektem Referenzpunkt.
    """
    cube_id = ids // 10  # Bestimmt den Würfel (Teamnummer)
    face_id = ids % 10   # Bestimmt die Seite des Würfels

    # Offset des Referenzpunkts (1.5 cm hinter dem Marker)
    offset = np.array([0, 0, 0.015], dtype=np.float32)

    # Berechne die neue Position des Würfels
    cube_position = tvecs.flatten() + offset

    # Position auf dem Würfel (Vorderseite, Rückseite, usw.)
    face_positions = {
        0: "Vorne",
        1: "Rechts",
        2: "Hinten",
        3: "Links",
        4: "Oben",
        5: "Unten"
    }

    # Yaw-Korrektur basierend auf der erkannten Seite
    face_yaw_correction = {
        0: 0,  # Vorderseite → 0°
        1: 90,  # Rechte Seite → 90°
        2: 180,  # Rückseite → 180°
        3: -90,  # Linke Seite → -90°
    }

    # Ursprüngliche Rotation des ArUco-Markers
    raw_yaw = float(rvecs.flatten()[2] * (180 / np.pi))

    # Endgültiger Yaw-Winkel des Würfels unter Berücksichtigung der Kameraansicht
    adjusted_yaw = raw_yaw + face_yaw_correction.get(face_id, 0)

    return {
        "cube_id": int(cube_id),
        "face": face_positions.get(face_id, "Unbekannt"),
        "Position": {
            "x": float(cube_position[0]),
            "y": float(cube_position[1]),
            "z": float(cube_position[2])
        },
        "Rotation": adjusted_yaw  # Korrigierter Yaw-Winkel
    }


def main():
    cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        print("Error: Cannot open video stream")
        exit()
    else:
        print("Success: Starting video stream")

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        corners, ids, _ = detector.detectMarkers(frame)

        if ids is not None:
            cubes = {}  # Dictionary zur Speicherung der Würfelpositionen

            aruco.drawDetectedMarkers(frame, corners, ids)

            for i in range(len(ids)):
                rvecs, tvecs = estimate_pose(corners[i], MARKER_SIZE, camera_matrix, dist_coeffs)

                if rvecs is not None and tvecs is not None:
                    cube_data = convert_marker_to_cube(int(ids[i][0]), tvecs, rvecs)

                    # Daten nach Cube-ID gruppieren
                    cube_id = cube_data["cube_id"]
                    if cube_id not in cubes:
                        cubes[cube_id] = {"id": cube_id, "faces": []}

                    cubes[cube_id]["faces"].append(cube_data)


                # Berechnung der Polarkoordinaten
                x, y = cube_data["Position"]["z"], cube_data["Position"]["x"]
                r = np.sqrt(x ** 2 + y ** 2)
                theta = np.arctan2(y, x) * (180 / np.pi)

                # Yaw-Winkel
                yaw = cube_data["Rotation"]

                # Senden der Werte über MQTT
                payload = {
                    "id": 4,
                    "Others": [
                        {
                            "id": cube_data["cube_id"],
                            "face": cube_data["face"],
                            "Position": {
                                "Distance": float(round(r, 3)),  # Abstand
                                "Angle": float(round(theta, 2)),  # Richtung
                                "Yaw": float(round(yaw, 2))  # Rotation
                                #"x": float(cube_data["Position"]["x"]),
                                #"y": float(cube_data["Position"]["y"]),
                                #"z": float(cube_data["Position"]["z"])
                            }
                        }
                    ]
                }

                # Senden der Würfel-Daten über MQTT
                for cube_id, data in cubes.items():
                    mqtt_payload = json.dumps(payload)
                    client.publish(MQTT_TOPIC_PUBLISH, mqtt_payload)

        cv2.imshow('ESP32 ArUco Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    client.loop_stop()
    client.disconnect()


if __name__ == '__main__':
    main()
