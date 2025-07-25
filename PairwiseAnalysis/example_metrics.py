import matplotlib.pyplot as plt
import numpy as np
import math

# Constants
bulk_density = 0.033  # Arbitrary example value
radius = 3.5  # Angstroms
inner_volume = (4/3) * math.pi * radius**3  # Volume of spherical region
regions = {
    "Low": 10,     # 10 particles
    "Medium": 50,  # 50 particles
    "High": 150    # 150 particles
}

colors = {"Low": "skyblue", "Medium": "seagreen", "High": "indianred"}

# Store data
density_ratios = []
pairwise_distances = []
counts = []

for label, n_particles in regions.items():
    counts.append(n_particles)
    # Density = count / volume / bulk_density
    local_density = n_particles / inner_volume
    density_ratios.append(local_density / bulk_density)

    # Generate synthetic particle positions in a cube (for easier spacing)
    np.random.seed(n_particles)  # ensure reproducibility
    positions = np.random.uniform(-radius, radius, size=(n_particles, 3))

    # Calculate pairwise distances
    distances = []
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            dist = np.linalg.norm(positions[i] - positions[j])
            distances.append(dist)
    pairwise_distances.append((label, distances))

# Plot 1: Relative Density
plt.figure(figsize=(6, 4))
plt.bar(regions.keys(), density_ratios, color=[colors[k] for k in regions])
plt.ylabel("Relative Density (vs Bulk)")
plt.title("Relative Density of Oxygens within Sphere")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("example_relative_density.png")
plt.close()

# Plot 2: Histogram of particle counts
plt.figure(figsize=(6, 4))
plt.bar(regions.keys(), counts, color=[colors[k] for k in regions])
plt.ylabel("Oxygens Near Center")
plt.title("Distribution of Oxygen Counts")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("example_count_distribution.png")
plt.close()

# Plot 3: Pairwise distances for each region
plt.figure(figsize=(8, 5))
for label, dists in pairwise_distances:
    plt.hist(dists, bins=20, alpha=0.6, label=f"{label} Density", color=colors[label], edgecolor='black')
plt.xlabel("Distance (Å)")
plt.ylabel("Frequency")
plt.title("Pairwise Distances Within Regions")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("example_pairwise_distances.png")
plt.close()
