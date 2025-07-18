import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import MDAnalysis as md
import math
from mpi4py import MPI
import csv

def index_to_xyz(index, x_bins, y_bins):
    z = index // (x_bins * y_bins)
    remainder = index % (x_bins * y_bins)
    y = remainder // x_bins
    x = remainder % x_bins
    return [x, y, z]

def center_of_box(index, x_bins, y_bins, bins_per_axis, box_size):
    boxcoords = index_to_xyz(index, x_bins, y_bins)
    return [((pos + 0.5) * (box_size / bins_per_axis)) for pos in boxcoords]

def distance3D(coord1, coord2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(coord1, coord2)))

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    PSF = 'w4096.psf'
    DCD = 'sample.dcd'
    sim = md.Universe(PSF, DCD)

    bisections = 1
    bins_per_axis = bisections + 1
    x_bins = bins_per_axis
    y_bins = bins_per_axis
    z_bins = bins_per_axis
    partitions = x_bins * y_bins * z_bins
    box_size = sim.trajectory[0].dimensions[0]
    partition_size = box_size / bins_per_axis
    radius_from_center = 3.5

    assert partition_size >= 2 * radius_from_center, (
        f"Partition size is {partition_size} cubic angstroms, which is less than 2r ({2 * radius_from_center})."
    )

    if rank == 0:
        print(f"Partition Size: {partition_size:.2f} Å")

    nframes = sim.trajectory.n_frames
    oxygens = sim.select_atoms('name OH2')

    particles_near_center = []

    for frame in range(nframes):
        sim.trajectory[frame]

        # Shift coordinates for safety
        for atom in oxygens:
            atom.position += box_size / 2

        boxes = [[] for _ in range(partitions)]
        for particle in oxygens.positions:
            xID = int((particle[0] / box_size) * x_bins)
            yID = int((particle[1] / box_size) * y_bins)
            zID = int((particle[2] / box_size) * z_bins)

            xID = min(max(xID, 0), x_bins - 1)
            yID = min(max(yID, 0), y_bins - 1)
            zID = min(max(zID, 0), z_bins - 1)
            boxIndex = xID + yID * x_bins + zID * x_bins * y_bins
            boxes[boxIndex].append(particle)

        # Rank 0 splits the boxes
        if rank == 0:
            box_chunks = np.array_split(boxes, size)
        else:
            box_chunks = None

        local_boxes = comm.scatter(box_chunks, root=0)

        # Local offset per rank
        box_sizes = comm.allgather(len(local_boxes))  # each rank tells how many boxes it got
        local_offset = sum(box_sizes[:rank])

        local_counts = []
        for i, box in enumerate(local_boxes):
            global_index = local_offset + i
            center = center_of_box(global_index, x_bins, y_bins, bins_per_axis, box_size)
            count = sum(1 for particle in box if distance3D(particle, center) <= radius_from_center)
            local_counts.append(count)

        # Gather back to root
        frame_counts = comm.gather(local_counts, root=0)

        if rank == 0:
            flat = [c for rank_result in frame_counts for c in rank_result]
            particles_near_center.append(flat)

    # Rank 0 plots results
    if rank == 0:
        flat_counts = [c for frame in particles_near_center for c in frame]

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
        ax.set_title("Distribution of Oxygens Near Partition Centers", fontsize=14)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        for count, x in zip(n, bins[:-1]):
            if count > 0:
                ax.text(x + (bins[1] - bins[0]) / 2, count, str(int(count)), ha='center', va='bottom', fontsize=10)
        plt.savefig("WaterHistogram.png", format='png')
        plt.show()
        with open("particles_near_center.csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Frame", *[f"Box{i}" for i in range(partitions)]])
            for frame_idx, frame in enumerate(particles_near_center):
                writer.writerow([frame_idx] + frame)