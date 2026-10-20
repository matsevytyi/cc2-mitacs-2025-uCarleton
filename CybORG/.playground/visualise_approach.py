import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

# Set seed for reproducibility
np.random.seed(42)


def generate_cluster_data(scenario_name):
    """Generates 3D points based on the updated 'status' and 'IP' scenarios."""
    if scenario_name == "status":
        # 1. Healthy (Dense Majority): 30 points, tightly clustered
        mean_healthy = [0, 0, 0]
        cov_healthy = np.eye(3) * 0.05  # Very low variance = high density
        healthy = np.random.multivariate_normal(mean_healthy, cov_healthy, 15)

        # 2. Restored (Sparse Minority): 7 points, scattered
        mean_restored = [3, 3, 2]
        cov_restored = np.eye(3) * 1.5  # High variance = sparse
        restored = np.random.multivariate_normal(mean_restored, cov_restored, 3)

        # 3. Compromised (Sparse Minority, different area): 3 points, sparse
        mean_compromised = [-3, 2, -2]
        cov_compromised = np.eye(3) * 1.2  # High variance = sparse
        compromised = np.random.multivariate_normal(
            mean_compromised, cov_compromised, 2
        )

        return {
            "healthy": healthy,
            "restored": restored,
            "compromised": compromised,
        }

    elif scenario_name == "IP":
        # Define explicit 3D positions for network gateways to build a graph backbone
        network_centers = {
            "Network 1 (192.168.1.2-128)": np.array([0.0, 0.0, 0.0]),
            "Network 2 (192.168.1.128-254)": np.array([2.5, 2.0, 0.5]),  # Close (shares 192.168.1)
            "Network 3 (192.168.2.x)": np.array([-4.0, -1.0, -2.5]),  # Farther away subnet
        }

        # Structure to track graph edges (connections) for the plotter
        connections = {
            "backbone": [
                # Connect Net 1 to Net 2, and Net 1 to Net 3
                (
                    network_centers["Network 1 (192.168.1.2-128)"],
                    network_centers["Network 2 (192.168.1.128-254)"],
                ),
                (
                    network_centers["Network 1 (192.168.1.2-128)"],
                    network_centers["Network 3 (192.168.2.x)"],
                ),
            ],
            "host_links": [],
        }

        def embed_ip_to_graph(ip_str, net_center):
            """Embeds an IP address as a node branching out from its gateway.

            Uses octet weights to establish distance and angular distribution
            to prevent straight lines, mimicking a star-topology graph structure.
            """
            octets = [int(o) for o in ip_str.split(".")]
            w = [8, 4, 2, 1]

            # Calculate a hierarchical proximity metric using your weight rule
            hierarchy_val = (
                octets[0] * w[0]
                + octets[1] * w[1]
                + octets[2] * w[2]
                + octets[3] * w[3]
            )

            # Determine radius: variation in O4 dictates how far it orbits from the center
            radius = 0.4 + (hierarchy_val % 30) / 20.0

            # Use deterministic pseudo-random angles based on O4 to fan out points
            # This shatters the straight lines into a natural graph cluster cloud
            phi = (octets[3] * 17) % 360 * (np.pi / 180)  # Azimuth angle
            theta = (octets[3] * 11) % 180 * (np.pi / 180)  # Elevation angle

            # Convert spherical coordinates to 3D Cartesian offsets
            offset_x = radius * np.sin(theta) * np.cos(phi)
            offset_y = radius * np.sin(theta) * np.sin(phi)
            offset_z = radius * np.cos(theta)

            host_coords = net_center + np.array([offset_x, offset_y, offset_z])

            # Append the connection line from gateway to this specific host
            connections["host_links"].append((net_center, host_coords))

            return host_coords

        # Simulating 3 interconnected networks (40 total points)
        net1_ips = [
            f"192.168.1.{i}"
            for i in np.random.choice(range(2, 32), 9, replace=False)
        ]
        net2_ips = [
            f"192.168.1.{i}"
            for i in np.random.choice(range(128, 158), 7, replace=False)
        ]
        net3_ips = [
            f"192.168.2.{i}"
            for i in np.random.choice(range(2, 32), 4, replace=False)
        ]

        net1_points = np.array(
            [
                embed_ip_to_graph(ip, network_centers["Network 1 (192.168.1.2-128)"])
                for ip in net1_ips
            ]
        )
        net2_points = np.array(
            [
                embed_ip_to_graph(
                    ip, network_centers["Network 2 (192.168.1.128-254)"]
                )
                for ip in net2_ips
            ]
        )
        net3_points = np.array(
            [
                embed_ip_to_graph(ip, network_centers["Network 3 (192.168.2.x)"])
                for ip in net3_ips
            ]
        )

        return {
            "Network 1 (192.168.1.2-128)": net1_points,
            "Network 2 (192.168.1.128-254)": net2_points,
            "Network 3 (192.168.2.x)": net3_points,
            "__meta_connections__": connections,
            "__meta_centers__": network_centers,
        }

    else:
        raise ValueError(f"Unknown scenario name: {scenario_name}")


# --- Configuration ---
active_scenario = "IP"
# active_scenario = "status"

# Generate data using the scenario function
data_groups = generate_cluster_data(active_scenario)

# Create 3D figure and axis
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(projection="3d")

# Plot logic adapted to dynamically handle multi-class dictionary output
if active_scenario == "status":
    ax.scatter(
        data_groups["healthy"][:, 0],
        data_groups["healthy"][:, 1],
        data_groups["healthy"][:, 2],
        c="green",
        alpha=0.8,
        s=201,
        # # label="Healthy Devices",
    )
    ax.scatter(
        data_groups["restored"][:, 0],
        data_groups["restored"][:, 1],
        data_groups["restored"][:, 2],
        c="red",
        alpha=0.7,
        s=201,
        # # label="Restored Devices",
    )
    ax.scatter(
        data_groups["compromised"][:, 0],
        data_groups["compromised"][:, 1],
        data_groups["compromised"][:, 2],
        c="orange",
        alpha=0.7,
        s=201,
        # label="Compromised Devices",
    )

elif active_scenario == "IP":
    colors = ["royalblue", "green", "teal"]

    # Extract metadata blocks and drop them so they don't break loop plotting
    connections = data_groups.pop("__meta_connections__")
    centers = data_groups.pop("__meta_centers__")

    # 1. Render Graph Lines (Backbone links between main networks)
    for start, end in connections["backbone"]:
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            c="black",
            linestyle="--",
            linewidth=2.5,
            alpha=0.7,
            zorder=1,
        )

    # 2. Render Graph Lines (Local links from gateways to hosts)
    for start, end in connections["host_links"]:
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            c="gray",
            linestyle="-",
            linewidth=1.7,
            alpha=0.25,
            zorder=1,
        )

    # 3. Render Router/Gateway Center Nodes
    # for net_name, center_coords in centers.items():
    #     ax.scatter(
    #         center_coords[0],
    #         center_coords[1],
    #         center_coords[2],
    #         c="black",
    #         marker="D",
    #         s=90,
    #         zorder=5,
    #         # label=f"Gateway ({net_name.split()[1]})",
    #     )

    # 4. Render Host Nodes
    for i, (net_name, points) in enumerate(data_groups.items()):
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=colors[i],
            alpha=0.85,
            s=201,
            zorder=3,
            # label=net_name,
        )

# ax.grid(True)
# ax.legend(loc="upper left")

ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

# Set initial camera point-of-view
ax.view_init(elev=15, azim=30)

# --- UI Button Setup for Saving ---
ax_button = plt.axes([0.8, 0.02, 0.15, 0.05])
btn_save = Button(ax_button, "Save PNG", color="lightgray", hovercolor="0.9")


def save_callback(event):
    filename = f"{active_scenario}.png"
    ax_button.set_visible(False)
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    print(f"Success: Plot saved as '{os.path.abspath(filename)}'")
    ax_button.set_visible(True)
    plt.draw()


btn_save.on_clicked(save_callback)

# Show the interactive plot window
plt.show()
