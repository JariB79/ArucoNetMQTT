import cv2
import cv2.aruco as aruco
import numpy as np
import requests
import paho.mqtt.client as mqtt
import json
import camera
import aruco_utils
import time
import threading


############################# MQTT ######################################
# Callback function for received MQTT messages
def on_message(client, userdata, msg):
    payload = json.loads(msg.payload.decode())
    cam_id = payload["id"]
    others = payload["Others"]

    update_camera_data(cam_id, others)


# MQTT settings
MQTT_BROKER = "192.168.178.37" # Waltenhofen: 192.168.0.252
MQTT_PORT = 1883
MQTT_TOPIC_PUBLISH = "EZS/beschtegruppe/4"
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
client.on_message = on_message
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.subscribe(MQTT_TOPICS_SUBSCRIBE)
#####################################################################


##################### ESP32-CAM Configuration #######################
ip_address = "192.168.178.65"  # ESP32-CAM IP address 192.168.0.156
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

lock = threading.Lock()
camera_data = []  # List of camera entries with camera ID, marker ids and timestamp


# updates camera_data
def update_camera_data(camera_id, others_list):

    with lock:
        camera_entry = next((entry for entry in camera_data if entry["id"] == camera_id), None)

        # Checks if there is already an entry for this camera
        if camera_entry is None:
            camera_entry = {
                "id": camera_id,
                "Others": others_list,
                "time": time.time()
            }
            camera_data.append(camera_entry)
        else:
            camera_entry["Others"] = others_list
            camera_entry["time"] = time.time()

        # write to JSON file
        try:
            with open("marker_positions.json", "w") as f:
                json.dump(camera_data, f, indent=2)
        except Exception as e:
            print(f"Error while writing to JSON file: {e}")



def main():
    # Starts the background thread for processing incoming MQTT messages
    client.loop_start()

    cap = cv2.VideoCapture(url)

    if cap.isOpened():
        print("Success: Starting video stream")
    else:
        print("Error: Cannot open video stream")
        exit()

    detector = aruco_utils.ger_aruco_detector()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Detect ArUco markers in frame and extract their corners and IDs
        corners, ids, _ = detector.detectMarkers(frame)

        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)

            cubes = {}  # Dictionary for storing the cube positions
            # payload stores rotation and translation vectors of the detected markers
            payload = {
                "id": 4,
                "Others": [],
                "time": time.time()
            }

            for marker_id, marker_corners in zip(ids.flatten(), corners):
                # rvecs represents the rotation vector in the form [rx, ry, rz], given in radians
                # tvecs represents the translation vector in the form [tx, ty, tz], given in meters
                rvecs, tvecs = aruco_utils.estimate_pose(marker_corners, MARKER_SIZE, camera_matrix, dist_coeffs)
                # print("rvecs: ", rvecs.flatten() * 180 / np.pi)
                # print("tvecs: ", tvecs.flatten())

                # x-axis: red, y-axis: green, z-axis: blue
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs, tvecs, 0.02)

                cube_data = aruco_utils.convert_marker_to_cube(marker_id, tvecs, rvecs)

                if rvecs is not None and tvecs is not None:

                    detected_marker = {
                        "detected_id": int(marker_id),
                        "Position": [
                            {"rvecs": rvecs.flatten().tolist()},
                            {"tvecs": tvecs.flatten().tolist()}
                        ]
                    }
                    payload["Others"].append(detected_marker)


                    # Group data by cube ID
                    cube_id = cube_data["cube_id"]
                    if cube_id not in cubes:
                        cubes[cube_id] = {"id": cube_id, "faces": [], "polar_coords": []}
                    cubes[cube_id]["faces"].append(cube_data)

                    # Calculate polar coordinates of the cube to display them in a coordinate system
                    #polar_data = aruco_utils.calc_polar_coordinates(cube_data)
                    #cubes[cube_id]["polar_coords"].append(polar_data)

            #for cube_id, cube_data in cubes.items():
            #    print(f"Cube-ID: {cube_id}")
            #    print(json.dumps(cube_data, indent=4))

            
            # Send values via MQTT
            try:
                mqtt_payload = json.dumps(payload)
                client.publish(MQTT_TOPIC_PUBLISH, mqtt_payload)
            except Exception as e:
                print(f"Failed to publish MQTT message: {e}")

            with lock:
                update_camera_data(payload["id"], payload["Others"])


        cv2.imshow('ESP32 ArUco Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    client.loop_stop()
    client.disconnect()


if __name__ == '__main__':
    main()
