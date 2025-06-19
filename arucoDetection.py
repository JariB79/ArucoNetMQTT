import cv2
import cv2.aruco as aruco
import numpy as np
import requests
import paho.mqtt.client as mqtt
import json
import camera
import aruco_utils
import time


############################# MQTT ######################################
# Callback function for received MQTT messages

def on_message(client, userdata, msg):
    print(f"Received message: {msg.topic}: {msg.payload.decode()}")
'''
# MQTT settings
MQTT_BROKER = "192.168.2.112" # 192.168.0.252
MQTT_PORT = 1883
MQTT_TOPIC_PUBLISH = "EZS/beschtegruppe/#4"
MQTT_TOPIC_SUBSCRIBE = "aruco/detection"
'''

MQTT_BROKER = "test.mosquitto.org"
MQTT_PORT = 1883
MQTT_TOPIC_PUBLISH = "EZS/beschtegruppe/4"
MQTT_TOPIC_SUBSCRIBE = "EZS/beschtegruppe/5"

# Initialise MQTT-Client
client = mqtt.Client()
client.on_message = on_message
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.subscribe(MQTT_TOPIC_SUBSCRIBE)
#####################################################################


##################### ESP32-CAM Configuration #######################
ip_address = "192.168.2.113"  # ESP32-CAM IP address 192.168.0.156
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



    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, _ = detector.detectMarkers(frame)
        # Detect ArUco markers in frame and extract their corners and IDs
        corners, ids, _ = detector.detectMarkers(frame)

        if ids is not None:
            cubes = {}  # Dictionary for storing the cube positions

            # payload stores rotation and translation vectors of the detected markers
            payload = {
                "id": 1,
                "Others": [],
                "time": time.time()
            }
            aruco.drawDetectedMarkers(frame, corners, ids)


            for marker_id, marker_corners in zip(ids.flatten(), corners):
                # rvecs represents the rotation vector in the form [rx, ry, rz], given in radians
                # tvecs represents the translation vector in the form [tx, ty, tz], given in meters
                # rvecs, tvecs = aruco_utils.estimate_pose(marker_corners, MARKER_SIZE, camera_matrix, dist_coeffs)
                ids, rvecs, tvecs = aruco_utils.get_aruco_markers(frame, camera_matrix, MARKER_SIZE, dist_coeffs)
                # x-axis: red, y-axis: green, z-axis: blau
                #cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs, tvecs, 0.02)


                #cube_data = aruco_utils.convert_marker_to_cube(marker_id, tvecs, rvecs)
                #print("-------------------------------------------")
                #print("rvecs (deg): ", rvecs * (180/np.pi) )
                #print("tvecs: ", tvecs)
                #print("-------------------------------------------")
                #print(cube_data)
                #print(aruco_utils.estimate_camera_position_from_tvec(tvecs, marker_id, marker_global_position=(0.0, 0.0, 0.0)))

                if rvecs is not None and tvecs is not None:
                    detected_marker = {
                        "detected_id": int(marker_id),
                        "Position": [
                            {"rvecs": rvecs[0].tolist()},
                            {"tvecs": tvecs[0].tolist()}
                        ]
                    }
                    payload["Others"].append(detected_marker)


                    # Group data by cube ID
                    cube_id = cube_data["cube_id"]
                    if cube_id not in cubes:
                        cubes[cube_id] = {"id": cube_id, "faces": [], "polar_coords": []}
                    cubes[cube_id]["faces"].append(cube_data)

                    # Calculate polar coordinates of the cube to display them in a coordinate system
                    polar_data = aruco_utils.calc_polar_coordinates(cube_data)
                    cubes[cube_id]["polar_coords"].append(polar_data)
                

            for cube_id, cube_data in cubes.items():
                print(f"Würfel-ID: {cube_id}")
                print(json.dumps(cube_data, indent=4))


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
