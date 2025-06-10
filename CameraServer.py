import cv2

from Echtzeit.MarkerDetection import ArucoDetection
from FASim.analysis.evaluate import frame

# Replace with the IP address of your ESP32
ip_address = '172.20.10.2'
url = f'http://{ip_address}:81/stream'

def main():
    # Open video stream
    cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        print("Error: Cannot open video stream")
        exit()
    else:
        print("Success: Starting video stream")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Show the frame
        cv2.imshow('ESP32 Stream', frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()
    ArucoDetecter = ArucoDetection()
    ArucoDetecter.detectInImage(frame)
    ArucoDetecter.showImage()


class ArucoDetection:
    def __init__(self):
        self.aruco_dict = aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.aruco_dict, parameters)

    def detectInImage(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.detector.detectMarkers(gray)

        return corners, ids, rejected, image

    def showImage(self, corners, ids, image):
        aruco.drawDetectedMarkers(image, corners, ids)
        cv2.imshow("ArUco Marker Detection", image)

    def __del__(self):
        cv2.destroyAllWindows()