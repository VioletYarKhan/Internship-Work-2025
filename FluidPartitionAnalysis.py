import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import MDAnalysis as md
import math
import string, sys, os
from mpi4py import MPI

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Only rank 0 loads the full simulation
if rank == 0:
    PSF = r'w4096.psf'
    DCD = r'sample.dcd'
    sim = md.Universe(PSF, DCD)
    nframes = sim.trajectory.n_frames
    box_size = sim.trajectory[0].dimensions[0]
    oxygens = sim.select_atoms('name OH2')
    oxy_positions_per_frame = []
    for ts in sim.trajectory:
        positions = oxygens.positions.copy()
        positions += box_size / 2  # adjust
        oxy_positions_per_frame.append(positions)
else:
    oxy_positions_per_frame = None
    nframes = None
    box_size = None

nframes = comm.bcast(nframes, root=0)
box_size = comm.bcast(box_size, root=0)
oxy_positions_per_frame = comm.bcast(oxy_positions_per_frame, root=0)

# Binning setup
bisections = 1
bins_per_axis = bisections + 1
x_bins = bins_per_axis
y_bins = bins_per_axis
z_bins = bins_per_axis
partitions = x_bins * y_bins * z_bins
partition_size = box_size / bins_per_axis
radius_from_center = 3.5

assert partition_size >= 2 * radius_from_center, (
    f"Partition size is {partition_size} < 2r = {2 * radius_from_center}. Adjust parameters."
)

def index_to_xyz(index):
    z = index // (x_bins * y_bins)
    remainder = index % (x_bins * y_bins)
    y = remainder // x_bins
    x = remainder % x_bins
    return [x, y, z]

def center_of_box(index):
    boxcoords = index_to_xyz(index)
    return [((i / bins_per_axis) * box_size + (box_size / bins_per_axis) / 2) for i in boxcoords]

def distance3D(coord1, coord2):
    return math.sqrt(sum((coord1[i] - coord2[i]) ** 2 for i in range(3)))


# Per-frame computation: each rank processes a subset of boxes per frame
local_particles_near_center = []

for frame in range(nframes):
    boxes = [[] for _ in range(partitions)]
    for particle in oxygens.atoms.positions:
        xID = int((particle[0] / box_size) / (1 / (x_bins)))
        yID = int((particle[1] / box_size) / (1 / (y_bins)))
        zID = int((particle[2] / box_size) / (1 / (z_bins)))

        xID = min(max(xID, 0), x_bins - 1)
        yID = min(max(yID, 0), y_bins - 1)
        zID = min(max(zID, 0), z_bins - 1)
        boxIndex = xID + yID * x_bins + zID * x_bins * y_bins
        boxes[boxIndex].append(particle)
    frame_counts = []
    for i, box in enumerate(boxes):
        center = center_of_box(i)
        count = 0
        for particle in box:
            if distance3D((particle[0], particle[1], particle[2]), center) <= radius_from_center:
                count += 1
        frame_counts.append(count)
    local_particles_near_center.append(frame_counts)

# Gather results from all ranks
particles_near_center = []
for frame_id in range(nframes):
    frame_data = comm.allreduce(local_particles_near_center[frame_id], op=MPI.SUM)
    if rank == 0:
        particles_near_center.append(frame_data)

# Plot on root only
if rank == 0:
    # Flatten for histogram
    flat_counts = [count for frame in particles_near_center for count in frame]

    fig, ax = plt.subplots(figsize=(8, 5), tight_layout=True)
    n, bins, patches = ax.hist(
        flat_counts,
        bins='auto',
        color='#4a90e2',
        edgecolor='black',
        alpha=0.85
    )
    ax.set_xlabel(f"Oxygens within {radius_from_center} Å of partition center", fontsize=12)
    ax.set_ylabel("Number of partitions", fontsize=12)
    ax.set_title(f"Distribution of Oxygens Near Partition Centers\n(within {radius_from_center} Å)", fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    for count, x in zip(n, bins[:-1]):
        if count > 0:
            ax.text(x + (bins[1] - bins[0]) / 2, count, str(int(count)), ha='center', va='bottom', fontsize=10)
    output_path = os.path.join(os.environ.get("SLURM_SUBMIT_DIR", "."), "Histogram.png")
    plt.savefig(output_path)
    print(f"Plot saved to: {output_path}")
    plt.show()
