import time
from datetime import datetime
import paho.mqtt.client as mqtt
import json
import threading
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from NodeVisualizer import NodeVisualizer


# MQTT Broker Setup
MQTT_BROKER = "test.mosquitto.org"
MQTT_PORT = 1883
MQTT_TOPIC = "EZS/beschtegruppe/4"


class NodeManager:
    """
    Verwaltet den Zustand der Knoten, relative Transformationen basierend auf Zeitstempeln
    und berechnet die globalen Koordinaten unter Berücksichtigung der erkannten Seite.
    """

    def __init__(self, anchor_cube_id=0):
        self.anchor_cube_id = anchor_cube_id
        self.nodes = {}
        # Format: (sender, detected) -> {'transform': matrix, 'timestamp': int}
        self.relative_transforms = {}
        self.lock = threading.Lock()  # Lock für Thread-sicheren Zugriff
        self.side_transforms = self._create_side_transforms()

    def _create_side_transforms(self):
        """
        Erstellt die 4x4-Transformationsmatrizen, um erkannte Seiten auf die Vorderseite zu mappen.
        Seite: 0 = vorne, 1 = rechts, 2 = hinten, 3 = links, 4 = oben, 5 = unten
        """
        transforms = {}
        euler_angles = {
            0: ('y', 180),
            1: ('y', 90),
            2: ('y', 0),
            3: ('y', -90),
            4: ('x', 90),
            5: ('x', -90)
        }

        for side_id, (axis, angle) in euler_angles.items():
            rot = Rotation.from_euler(axis, angle, degrees=True).as_matrix()
            mat = np.identity(4)
            mat[:3, :3] = rot
            transforms[side_id] = mat

        return transforms

    def get_camera_id(self, device_id):
        return device_id // 10

    def get_side_id(self, device_id):
        return device_id % 10

    def get_all_nodes(self):
        with self.lock:
            return self.nodes.copy()

    def _init_node_if_needed (self, cam_id):
        if cam_id not in self.nodes:
            self.nodes[cam_id] = {
                'is_anchored': False,
                'global_transform': np.identity(4),
                'is_anchor_node': False,
                'last_seen': time.time()  # Initialen Zeitstempel setzen
            }
            if cam_id == self.anchor_cube_id:
                self.nodes[cam_id]['is_anchored'] = True
                self.nodes[cam_id]['is_anchor_node'] = True

    def process_mqtt_message(self, payload):
        """
        Verarbeitet eine MQTT-Nachricht mit Marker-Detektionen.

        Wandelt die Markerpose (Marker→Kamera) in Kamera→Marker um,
        korrigiert die Ausrichtung basierend auf der erkannten Seite und speichert die
        Transformation zwischen Sender und Detektiertem – aber nur, wenn sie aktueller ist
        als eine bereits vorhandene. Aktualisiert anschließend alle globalen Positionen.
        """
        try:

            try:
                data = json.loads(payload)
                sender_id = data['id']
                raw_time = data.get("time")
                pt = datetime.strptime(raw_time, "%Y-%m-%d %H:%M:%S")
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                print(f" Ungültiger Payload oder Zeitangabe: {e} | Daten: {payload}")
                return


            message_timestamp = pt.second + pt.minute * 60 + pt.hour * 3600

            if message_timestamp is None: return

            with self.lock:
                current_time = time.time()
                for other in data.get('Others', []):
                    detected_id = other['detected_id']
                    detected_cam = self.get_camera_id(detected_id)
                    side_id = self.get_side_id(detected_id)

                    # Schützt vor Selbstbeobachtung
                    if sender_id == detected_cam: continue

                    if side_id not in self.side_transforms:
                        print(f"Warnung: Ungültige Seiten-ID {side_id} für {detected_id} erkannt. Ignoriere.")
                        continue

                    self._init_node_if_needed (sender_id)
                    self._init_node_if_needed (detected_cam)

                    # Zeitstempel für beide beteiligten Knoten aktualisieren
                    self.nodes[sender_id]['last_seen'] = current_time
                    self.nodes[detected_cam]['last_seen'] = current_time


                    existing_relation = self.relative_transforms.get((sender_id, detected_cam))
                    if existing_relation and message_timestamp <= existing_relation['timestamp']:
                        continue

                    '''
                    print("Aktuelle Relationen:")
                    for key, rel in self.relative_transforms.items():
                        print(f"{key}: Zeit {rel['timestamp']}, Matrix:\n{rel['transform']}")
                        
                    print("Knoten im Koordinatensystem:")
                    for cam_id, node in self.nodes.items():
                        print(
                            f"G{cam_id} | Anker: {node['is_anchored']} | Pos: {np.round(node['global_transform'][:3, 3], 2)}")
                    '''

                    position_data = other.get('Position', [])

                    rvecs, tvecs = None, None # Pose des MARKERS aus Sicht der KAMERA
                    for p in position_data:
                        if 'rvecs' in p: rvecs = np.array(p['rvecs'])
                        if 'tvecs' in p: tvecs = np.array(p['tvecs'])
                    if rvecs is None or tvecs is None: continue

                    rotation_matrix = Rotation.from_rotvec(rvecs).as_matrix()

                    measured_transform = np.identity(4)
                    measured_transform[:3, :3] = rotation_matrix
                    measured_transform[:3, 3] = tvecs.flatten()

                    side_correction_transform = self.side_transforms[side_id]


                    normalized_transform = measured_transform @ side_correction_transform

                    new_data = {'transform': normalized_transform, 'timestamp': message_timestamp}
                    inv_new_data = {'transform': np.linalg.inv(normalized_transform), 'timestamp': message_timestamp}

                    self.relative_transforms[(sender_id, detected_cam)] = new_data
                    self.relative_transforms[(detected_cam, sender_id)] = inv_new_data

                    self.update_all_positions()

        except (json.JSONDecodeError, KeyError, np.linalg.LinAlgError) as e:
            print(f"Fehler beim Verarbeiten der Nachricht oder bei der Berechnung: {e}")

    def remove_old_nodes(self, timeout=10):
        """
        Entfernt Knoten und deren Relationen, die länger als 'timeout' Sekunden
        nicht gesehen wurden. Der Ankerknoten wird nie entfernt.
        """
        with self.lock:
            current_time = time.time()
            nodes_to_remove = []
            for cam_id, node_data in self.nodes.items():
                if cam_id == self.anchor_cube_id:
                    continue
                if current_time - node_data.get('last_seen', 0) > timeout:
                    nodes_to_remove.append(cam_id)

            if not nodes_to_remove:
                return

            print(f"Entferne veraltete Knoten: {nodes_to_remove}")

            for cam_id in nodes_to_remove:
                if cam_id in self.nodes:
                    del self.nodes[cam_id]

            relations_to_remove = []
            for start_node, end_node in list(self.relative_transforms.keys()):
                if start_node in nodes_to_remove or end_node in nodes_to_remove:
                    relations_to_remove.append((start_node, end_node))

            for relation in relations_to_remove:
                if relation in self.relative_transforms:
                    del self.relative_transforms[relation]

            self.update_all_positions()

    def update_all_positions(self):
        """
        Berechnet die globalen Positionen aller Knoten neu, ausgehend vom Anker.
        Diese Methode ist für den internen Gebrauch und sollte innerhalb eines Locks aufgerufen werden.
        """

        if self.anchor_cube_id not in self.nodes: return

        for cam_id, node in self.nodes.items():
            if cam_id != self.anchor_cube_id:
                node['is_anchored'] = False

        self.nodes[self.anchor_cube_id]['global_transform'] = np.identity(4)
        self.nodes[self.anchor_cube_id]['is_anchored'] = True

        for i in range(len(self.nodes) + 1):
            updated_in_pass = False
            for (start_node, end_node), data in self.relative_transforms.items():
                if self.nodes.get(start_node, {}).get('is_anchored'):
                    T_rel = data['transform']
                    T_global_start = self.nodes[start_node]['global_transform']
                    T_new_global_end = T_global_start @ T_rel

                    if not self.nodes.get(end_node, {}).get('is_anchored') or not np.allclose(
                            self.nodes[end_node]['global_transform'], T_new_global_end):
                        self._init_node_if_needed (end_node)
                        self.nodes[end_node]['global_transform'] = T_new_global_end
                        self.nodes[end_node]['is_anchored'] = True
                        updated_in_pass = True

            if not updated_in_pass and i > 0:
                break


# --- MQTT Client-Funktionen ---

def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print(f"Verbindung mit MQTT Broker erfolgreich. Abonniere Topic: {MQTT_TOPIC}")
        client.subscribe(MQTT_TOPIC)
    else:
        print(f"Verbindung mit MQTT Broker fehlgeschlagen: {rc}")


def on_message(client, userdata, msg):
    node_manager = userdata['node_manager']
    node_manager.process_mqtt_message(msg.payload.decode())
    print("msg.payload.decode():", msg.payload.decode())


############################### MQTT-Client und Visualisierung starten ##################################
def main():

    node_manager = NodeManager(anchor_cube_id=0)
    visualizer = NodeVisualizer(node_manager, x_lims=(-1, 1), z_lims=(-1, 1))

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message

    client.user_data_set({'node_manager': node_manager})

    print(f"Stelle Verbindung zum MQTT-Broker her ({MQTT_BROKER}:{MQTT_PORT})...")
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
    except Exception as e:
        print(f"Keine Verbindung zu MQTT Broker möglich: {e}")
        return

    client.loop_start()
    visualizer.start_animation()
    plt.show()

    # Wird erst ausgeführt, nachdem das Plot-Fenster geschlossen wurde
    client.loop_stop()
    print("Programm wird beendet.")
#######################################################################################################



if __name__ == "__main__":
    main()