import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.colors as mcolors


class NodeVisualizer:
    """
    Visualisiert die Positionen aller Knoten in einer 2D-XZ-Ansicht mit Matplotlib.
    """

    def __init__(self, node_manager, x_lims=(-100, 100), z_lims=(-100, 100)):
        self.node_manager = node_manager
        self.fig, self.ax = plt.subplots(figsize=(10, 10))

        self.x_lims = x_lims
        self.z_lims = z_lims

        # Achsen-Layout einstellen (dunkles Design)
        self.ax.set_xlabel("Globale X-Position")
        self.ax.set_ylabel("Globale Z-Position")
        self.ax.set_title("Globale 2D-Karte der Nodes")
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.grid(True, color='gray')

        self.fig.patch.set_facecolor('black')
        self.ax.set_facecolor('black')

        self.ax.tick_params(colors='white')  # Achsenbeschriftung weiß
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.title.set_color('white')

    def _update_frame(self, frame):
        # Vor dem Zeichnen veraltete Knoten entfernen
        self.node_manager.remove_old_nodes(timeout=10)

        self.ax.clear()

        # Achsen erneut konfigurieren (nach ax.clear())
        self.ax.set_xlim(self.x_lims)
        self.ax.set_ylim(self.z_lims)
        self.ax.set_xlabel("Globale X-Position")
        self.ax.set_ylabel("Globale Z-Position")
        self.ax.set_title("Globale 2D-Karte der Kamerapositionen (Ursprung: Node 0, XZ-Ebene)")
        self.ax.grid(True, color='gray')
        self.ax.set_facecolor('black')
        self.ax.tick_params(colors='white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.title.set_color('white')

        nodes = self.node_manager.get_all_nodes()

        # Farbpalette definieren
        id_list = list(nodes.keys())
        num_ids = len(id_list)
        cmap = cm.get_cmap('hsv', num_ids)
        color_map = {cam_id: mcolors.to_hex(cmap(i)) for i, cam_id in enumerate(id_list)}

        for cam_id, node in nodes.items():
            if node['is_anchored']:
                global_transform = node['global_transform']
                pos = global_transform[:3, 3]
                is_anchor = node.get('is_anchor_node', False)

                color = 'white' if is_anchor else color_map[cam_id]
                label = f"Cube {cam_id} (Ursprung)" if is_anchor else f"Kamera {cam_id}"

                # Punkt plotten
                self.ax.plot(pos[0], pos[2], 'o', color=color, markersize=12, label=label)

                # ID und Name direkt an Position anzeigen
                self.ax.text(pos[0] + 1.5, pos[2] + 1.5, f"{label}", fontsize=8, color=color)

                # Richtungspfeil
                direction_vec = global_transform[:3, 2]
                self.ax.arrow(pos[0], pos[2], direction_vec[0] * 0.15, direction_vec[2] * 0.15,
                              head_width=0.03, head_length=0.04, width=0.01,
                              fc='yellow', ec='black', alpha=0.6)
                # Pfeilrichtung
                dx, dz = direction_vec[0] * 0.15, direction_vec[2] * 0.15
                # Text direkt am Pfeilkopf anzeigen (leicht versetzt)
                arrow_tip_x = pos[0] + dx
                arrow_tip_z = pos[2] + dz
                self.ax.text(arrow_tip_x + 0.01, arrow_tip_z + 0.01,
                             f"cam: {cam_id}", fontsize=8, color='cyan')

        handles, labels = self.ax.get_legend_handles_labels()
        if labels:
            by_label = dict(zip(labels, handles))
            legend = self.ax.legend(by_label.values(), by_label.keys())
            if legend:
                legend.get_frame().set_facecolor('black')
                legend.get_frame().set_edgecolor('white')
                for text in legend.get_texts():
                    text.set_color('white')

        self.ax.set_aspect('equal', adjustable='box')

    def start_animation(self):
        # FuncAnimation aktualisiert die Figur im Intervall von 500ms
        self.ani = animation.FuncAnimation(self.fig, self._update_frame, interval=500, blit=False)
