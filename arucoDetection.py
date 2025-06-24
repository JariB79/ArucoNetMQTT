import cv2
import cv2.aruco as aruco
import requests
import paho.mqtt.client as mqtt
import json
import camera
import aruco_utils
import time
from datetime import datetime


############################# MQTT ######################################
# Callback function for received MQTT messages
'''
def on_message(client, userdata, msg):
    payload = json.loads(msg.payload.decode())
    cam_id = payload["id"]
    others = payload["Others"]

    update_camera_data(cam_id, others)
'''

# MQTT settings
MQTT_BROKER = "test.mosquitto.org" # Waltenhofen: 192.168.0.252 test.mosquitto.org
MQTT_PORT = 1883
MQTT_TOPIC_PUBLISH = "EZS/beschtegruppe1/4"
MQTT_TOPICS_SUBSCRIBE = [
    ("EZS/beschtegruppe/1", 0),
    ("EZS/beschtegruppe/2", 0),
    ("EZS/beschtegruppe/3", 0),
    ("EZS/beschtegruppe/4", 0),
    ("EZS/beschtegruppe/5", 0),
    ("EZS/beschtegruppe/6", 0),
    ("EZS/beschtegruppe/7", 0),
]

'''
MQTT_BROKER = "test.mosquitto.org"
MQTT_PORT = 1883
MQTT_TOPIC_PUBLISH = "EZS/beschtegruppe/4"
MQTT_TOPIC_SUBSCRIBE = "EZS/beschtegruppe/#"
'''

# Initialise MQTT-Client
client = mqtt.Client()
#client.on_message = on_message
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.subscribe(MQTT_TOPICS_SUBSCRIBE)
#####################################################################


##################### ESP32-CAM Configuration #######################
ip_address = "192.168.0.156"  # ESP32-CAM IP address 192.168.0.156
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
    # Starts the background thread for processing incoming MQTT messages
    client.loop_start()

    cap = cv2.VideoCapture(url)

    if cap.isOpened():
        print("Success: Starting video stream")
    else:
        print("Error: Cannot open video stream")
        exit()

    detector = aruco_utils.get_aruco_detector()
    last_send_time = 0
    send_interval = 0.5  # 500 ms

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Detect ArUco markers in frame and extract their corners and IDs
        corners, ids, _ = detector.detectMarkers(frame)
        current_time = time.time()
        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)

            if current_time - last_send_time >= send_interval:
                last_send_time = current_time
                cubes = {}  # Dictionary for storing the cube positions
                # payload stores rotation and translation vectors of the detected markers
                payload = {
                    "id": 4,
                    "Others": [],
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

                for marker_id, marker_corners in zip(ids.flatten(), corners):
                    rvecs, tvecs = aruco_utils.estimate_pose(marker_corners, MARKER_SIZE, camera_matrix, dist_coeffs)

                    # x-axis: red, y-axis: green, z-axis: blue
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs, tvecs, 0.02)

                    if rvecs is not None and tvecs is not None:
                        detected_marker = {
                            "detected_id": int(marker_id),
                            "Position": [
                                {"rvecs": rvecs.flatten().tolist()},
                                {"tvecs": tvecs.flatten().tolist()}
                            ]
                        }
                        payload["Others"].append(detected_marker)

                # Send values via MQTT
                try:
                    mqtt_payload = json.dumps(payload)
                    client.publish(MQTT_TOPIC_PUBLISH, mqtt_payload)
                except Exception as e:
                    print(f"Failed to publish MQTT message: {e}")


        cv2.imshow('ESP32 ArUco Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    client.loop_stop()
    client.disconnect()


if __name__ == '__main__':
    main()
