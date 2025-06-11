import cv2
import cv2.aruco as aruco
import numpy as np
import requests
import paho.mqtt.client as mqtt


# 3D-Koordinaten der Marker-Eckpunkte (bei flachem Marker auf der XY-Ebene)
def get_marker_3d_points(marker_size):
    half_size = marker_size / 2
    return np.array([
        [-half_size, half_size, 0],
        [half_size, half_size, 0],
        [half_size, -half_size, 0],
        [-half_size, -half_size, 0]
    ], dtype=np.float32)


# Berechnung der Pose mit `solvePnP`
def estimate_pose(corners, marker_size, camera_matrix, dist_coeffs):
    marker_3d_points = get_marker_3d_points(marker_size)
    success, rvecs, tvecs = cv2.solvePnP(marker_3d_points, corners[0], camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE)

    if success:
        return np.array(rvecs, dtype=np.float32), np.array(tvecs, dtype=np.float32)
    else:
        return None, None


# MQTT Einstellungen
MQTT_BROKER = "192.168.0.252"
MQTT_PORT = 1883
MQTT_TOPIC_PUBLISH = "$SYS/aruco/detection"
MQTT_TOPIC_SUBSCRIBE = "aruco/config"

# Callback-Funktion für empfangene MQTT-Nachrichten
def on_message(client, userdata, msg):
    print(f"Empfangene Nachricht: {msg.topic}: {msg.payload.decode()}")

# Initialisiere MQTT-Client
client = mqtt.Client()
client.on_message = on_message
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.subscribe(MQTT_TOPIC_SUBSCRIBE)
client.loop_start()


# Marker-Größe in Metern
marker_size = 0.025

# ESP32-CAM IP-Adresse
ip_address = "192.168.0.156"

# URL für das Setzen der Auflösung auf UXGA (1600x1200)
resolution_url = (f"http://{ip_address}/control?var=framesize&val=10")

# Anfrage senden, um die Auflösung zu ändern
response = requests.get(resolution_url)

if response.status_code == 200:
    print("Auflösung erfolgreich auf UXGA (1600x1200) gesetzt!")
else:
    print("Fehler beim Setzen der Auflösung:", response.status_code)

url = f'http://{ip_address}:81/stream?res=UXGA'

# Variablen für Kamerakalibrierung
fx, fy, cx, cy = 299.6479, 307.0981, 161.4847, 126.4022
k1, k2, p1, p2, k3 = -0.0657, 0.4584, 0, 0, 0

camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)


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
            aruco.drawDetectedMarkers(frame, corners, ids)

            for i in range(len(ids)):
                rvecs, tvecs = estimate_pose(corners[i], marker_size, camera_matrix, dist_coeffs)

                if rvecs is not None and tvecs is not None:
                    if rvecs.size == 3 and tvecs.size == 3:
                        rvecs = rvecs.reshape((3, 1))
                        tvecs = tvecs.reshape((3, 1))

                        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs, tvecs, marker_size)
                    else:
                        print("Warnung: Unerwartete Pose-Schätzung!")

                # Berechnung der Polarkoordinaten
                x, y = tvecs.flatten()[2], tvecs.flatten()[0]
                r = np.sqrt(x ** 2 + y ** 2)
                theta = np.arctan2(y, x) * (180 / np.pi)

                # Yaw-Winkel
                yaw = rvecs.flatten()[2] * (180 / np.pi)

                # Senden der Werte über MQTT
                payload = f"Marker {ids[i][0]} - r: {r:.3f} m, θ: {theta:.2f}°, Yaw: {yaw:.2f}°"
                client.publish(MQTT_TOPIC_PUBLISH, payload)

        cv2.imshow('ESP32 ArUco Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    client.loop_stop()
    client.disconnect()


if __name__ == '__main__':
    main()
