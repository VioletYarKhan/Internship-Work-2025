import matplotlib.pyplot as plt
import numpy as np
import math

# Parameters
radius = 3.5  # Radius of sampling sphere
box_size = 20  # Size of the square box (20x20)
num_particles = 200  # Total particles uniformly distributed in the box
center = (box_size / 2, box_size / 2)

# Generate uniformly distributed particles
xs = np.random.uniform(0, box_size, num_particles)
ys = np.random.uniform(0, box_size, num_particles)
positions = np.column_stack((xs, ys))

# Determine which particles are within 3.5 of the center
distances_to_center = np.linalg.norm(positions - center, axis=1)
within_radius_mask = distances_to_center <= radius
within_radius_positions = positions[within_radius_mask]

# Count and density calculation
count_in_sphere = len(within_radius_positions)
area_sphere = math.pi * radius**2
area_box = box_size ** 2
bulk_density = num_particles / area_box
local_density = count_in_sphere / area_sphere
relative_density = local_density / bulk_density

# Plot setup
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, box_size)
ax.set_ylim(0, box_size)
ax.set_aspect("equal")
ax.set_title("Pairwise Distances and Density Example", fontsize=25)


# Draw all particles
oxs = []
oys = []
for i, pos in enumerate(positions):
    if (not within_radius_mask[i]):
        oxs.append(pos[0])
        oys.append(pos[1])

ax.scatter(oxs, oys, color="lightgray", label="Outside Radius", zorder=0)

# Draw center sphere
circle = plt.Circle(center, radius, color="blue", fill=False, linestyle="--", linewidth=2)
ax.add_patch(circle)

# Draw pairwise distances from each center particle to all others
for p1 in within_radius_positions:
    for p2 in positions:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='gray', alpha=0.05, zorder=5)

# Draw particles within center sphere
ax.scatter(*within_radius_positions.T, color="dodgerblue", edgecolor="black", label="Within Radius", zorder=10)

# Annotations
ax.text(1, box_size - 1, f"Count in Radius = {count_in_sphere}", fontsize=20)
ax.text(1, box_size - 2, f"Relative Density = {relative_density:.2f}", fontsize=20)

# ax.legend(loc = "lower left", fontsize=20)
plt.tight_layout()
plt.savefig("ExampleParams.png", dpi=500)
plt.show()
