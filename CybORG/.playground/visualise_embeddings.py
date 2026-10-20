import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button
# Requires scikit-learn: pip install scikit-learn
from sklearn.decomposition import PCA

# Set seed for reproducibility
np.random.seed(42)

def generate_high_dim_embeddings():
    """Simulates 64-dim embeddings per octet concatenated into a 256-dim vector.
    
    Returns high-dimensional vectors and labels for the 13 official CC2 hosts.
    """
    cc2_inventory = {
        "User Subnet (192.168.1.x)": [
            ("User0", "192.168.1.10"), ("User1", "192.168.1.11"),
            ("User2", "192.168.1.12"), ("User3", "192.168.1.13"),
            ("User4", "192.168.1.14")
        ],
        "Enterprise Subnet (192.168.2.x)": [
            ("Enterprise0", "192.168.2.50"), 
            ("Enterprise1", "192.168.2.51"),
            ("Enterprise2", "192.168.2.52")
        ],
        "Operational Subnet (192.168.3.x)": [
            ("Operational0", "192.168.3.100"), 
            ("Operational1", "192.168.3.101"),
            ("Operational2", "192.168.3.102"),
            ("Operator0",    "192.168.3.200"), 
            ("OpServer0",    "192.168.3.250")
        ]
    }

    all_vectors = []
    host_labels = []
    subnet_assignments = []

    # Helper to simulate 64-dimensional embedding for an octet value
    def get_octet_embedding(octet_value, octet_position):
        # Base signal is deterministic based on the octet value and its place
        base_signal = np.sin(np.linspace(0, np.pi, 64)) * (octet_value / 255.0)
        # Add high-dimensional structural features per octet type
        if octet_position == 0:  # O1
            base_signal += 2.0
        elif octet_position == 1:  # O2
            base_signal += 1.0
        # Add a tiny bit of Gaussian noise to simulate realistic continuous embeddings
        noise = np.random.normal(0, 0.05, 64)
        return base_signal + noise

    # Process all hosts and build the 256-dim concatenated array
    for subnet_idx, (subnet_name, hosts) in enumerate(cc2_inventory.items()):
        for name, ip in hosts:
            octets = [int(o) for o in ip.split(".")]
            
            # Generate 64-dim embeddings for each of the 4 octets
            o1_emb = get_octet_embedding(octets[0], 0)
            o2_emb = get_octet_embedding(octets[1], 1)
            o3_emb = get_octet_embedding(octets[2], 2)
            o4_emb = get_octet_embedding(octets[3], 3)
            
            # Concatenate 4 x 64-dim vectors into ONE 256-dimensional vector
            concatenated_vector = np.concatenate([o1_emb, o2_emb, o3_emb, o4_emb])
            
            all_vectors.append(concatenated_vector)
            host_labels.append(name)
            subnet_assignments.append(subnet_idx) # Save index for color mapping

    return np.array(all_vectors), host_labels, subnet_assignments


# --- Data Generation & Dimensionality Reduction ---
# 1. Get the 256-dimensional data matrix (Shape: 13 hosts x 256 features)
high_dim_data, labels, subnet_ids = generate_high_dim_embeddings()

# 2. Apply PCA to compress 256 dimensions down to 3 visualizable components
pca = PCA(n_components=3, random_state=42)
low_dim_data = pca.fit_transform(high_dim_data)

# Print explained variance to console to check embedding representation quality
print(f"PCA Summary: Explained variance ratio of the 3 axes: {pca.explained_variance_ratio_}")
print(f"Total variance captured in 3D: {sum(pca.explained_variance_ratio_)*100:.2f}%")


# --- Plotting Interactive Scene ---
fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(projection="3d")

subnet_colors = ["royalblue", "green", "teal"]
subnet_names = ["User Subnet", "Enterprise Subnet", "Operational Subnet"]

# Group data by subnet to create a clean plotting loop and accurate legend
for s_idx in range(3):
    indices = [i for i, x in enumerate(subnet_ids) if x == s_idx]
    
    # Extract X, Y, Z coordinates computed by PCA
    x_coords = low_dim_data[indices, 0]
    y_coords = low_dim_data[indices, 1]
    z_coords = low_dim_data[indices, 2]
    
    # Scatter plot the nodes
    ax.scatter(
        x_coords, y_coords, z_coords,
        c=subnet_colors[s_idx],
        s=80,
        alpha=0.9,
        edgecolors='black',
        linewidth=0.5,
        label=subnet_names[s_idx]
    )
    
    # Add textual labels near the points to see exactly where OpServer0 or User0 lands
    for idx in indices:
        ax.text(
            low_dim_data[idx, 0] + 0.05, 
            low_dim_data[idx, 1] + 0.05, 
            low_dim_data[idx, 2] + 0.05, 
            labels[idx], 
            fontsize=9,
            weight='bold' if "Server" in labels[idx] else 'normal'
        )

# Graph cosmetics
# ax.grid(True)
# ax.legend(loc="upper left")

ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

# Adjust initial point of view
ax.view_init(elev=25, azim=135)


# --- UI Button Setup for Saving ---
ax_button = plt.axes([0.8, 0.02, 0.15, 0.05])
btn_save = Button(ax_button, "Save PNG", color="lightgray", hovercolor="0.9")

def save_callback(event):
    filename = "high_dim_ip_embedding.png"
    ax_button.set_visible(False)
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    print(f"Success: High-dim plot saved as '{os.path.abspath(filename)}'")
    ax_button.set_visible(True)
    plt.draw()

btn_save.on_clicked(save_callback)

plt.show()
