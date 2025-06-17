import cv2
import cv2.aruco as aruco
import numpy as np
import requests
import paho.mqtt.client as mqtt
import json
import camera
import aruco_utils


############################# MQTT ######################################
# Callback function for received MQTT messages
def on_message(client, userdata, msg):
    print(f"Received message: {msg.topic}: {msg.payload.decode()}")

# MQTT settings
MQTT_BROKER = "192.168.0.252"
MQTT_PORT = 1883
MQTT_TOPIC_PUBLISH = "aruco/detection"
MQTT_TOPIC_SUBSCRIBE = "aruco/detection"

# Initialise MQTT-Client
client = mqtt.Client()
client.on_message = on_message
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.subscribe(MQTT_TOPIC_SUBSCRIBE)

# Starts the background thread for processing incoming MQTT messages
client.loop_start()
#####################################################################


##################### ESP32-CAM Configuration #######################
ip_address = "192.168.0.156"  # ESP32-CAM IP address
url = f'http://{ip_address}:81/stream'

# URL for setting resolution to VGA (640x480)
resolution_url = (f"http://{ip_address}/control?var=framesize&val=10")
response = requests.get(resolution_url)

if response.status_code == 200:
    print("Resolution successfully set to VGA (640x480)!")
else:
    print("Error when setting the resolution:", response.status_code)
#####################################################################


MARKER_SIZE = 0.020  # marker size in meters
camera_matrix = camera.get_camera_matrix()
dist_coeffs = camera.get_dist_coeffs()

def main():
    cap = cv2.VideoCapture(url)

    if cap.isOpened():
        print("Success: Starting video stream")
    else:
        print("Error: Cannot open video stream")
        exit()

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        # Detect ArUco markers in frame and extract their corners and IDs
        corners, ids, _ = detector.detectMarkers(frame)

        if ids is not None:
            cubes = {}  # Dictionary for storing the cube positions

            aruco.drawDetectedMarkers(frame, corners, ids)

            for i in range(len(ids)):
                rvecs, tvecs = aruco_utils.estimate_pose(corners[i], MARKER_SIZE, camera_matrix, dist_coeffs)

                if rvecs is not None and tvecs is not None:
                    cube_data = aruco_utils.convert_marker_to_cube(int(ids[i][0]), tvecs, rvecs)

                    # Group data by cube ID
                    cube_id = cube_data["cube_id"]
                    if cube_id not in cubes:
                        cubes[cube_id] = {"id": cube_id, "faces": []}

                    cubes[cube_id]["faces"].append(cube_data)


                # Calculate  polar coordinates
                x, y = cube_data["Position"]["z"], cube_data["Position"]["x"]
                r = np.sqrt(x ** 2 + y ** 2)
                theta = np.arctan2(y, x) * (180 / np.pi)

                # Yaw angle
                yaw = cube_data["Rotation"]

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

                # Send values via MQTT
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
