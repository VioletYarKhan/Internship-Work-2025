import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import MDAnalysis as md
import math
from mpi4py import MPI
import csv

def array_split(lst, num_splits):
    n = len(lst)
    quotient, remainder = divmod(n, num_splits)

    result = []
    start = 0
    for i in range(num_splits):
        end = start + quotient + (1 if i < remainder else 0)
        result.append(lst[start:end])
        start = end
    return result

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

    box_size = sim.trajectory[0].dimensions[0]
    partition_size_wanted = 15
    bins_per_axis = round(box_size/partition_size_wanted)
    # bins_per_axis = 2

    x_bins = bins_per_axis
    y_bins = bins_per_axis
    z_bins = bins_per_axis
    partitions = x_bins * y_bins * z_bins
    partition_size = box_size / bins_per_axis
    radius_from_center = 3


    assert partition_size >= 2 * radius_from_center, (f"Partition size is {partition_size} cubic angstroms, which is less than 2r ({2 * radius_from_center}).")

    if rank == 0:
        print(f'{bins_per_axis} bins per axis of {partition_size} cubic angstroms')

    nframes = sim.trajectory.n_frames
    oxygens = sim.select_atoms('name OH2')

    bulk_density = len(oxygens)/pow(box_size, 3)

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
            box_chunks = array_split(boxes, size)
        else:
            box_chunks = None

        
        local_boxes = comm.scatter(box_chunks, root=0)

        # Local offset per rank
        box_sizes = comm.allgather(len(local_boxes))  # each rank tells how many boxes it got
        local_offset = sum(box_sizes[:rank])

        local_counts = []
        pairwise_distances = []
        # print(f"Rank {rank} started boxes {local_offset}-{local_offset+len(local_boxes)} of frame {frame}")
        for i, box in enumerate(local_boxes):
            global_index = local_offset + i
            center = center_of_box(global_index, x_bins, y_bins, bins_per_axis, box_size)
            central_particles = []
            noncentral_particles = []
            for particle in box:
                if distance3D(particle, center) <= radius_from_center:
                    central_particles.append(particle)
                else:
                    noncentral_particles.append(particle)

            for j, cen_particle in enumerate(central_particles):
                for particle in noncentral_particles:
                    pairwise_distances.append(distance3D(particle, cen_particle))
                for other_central in central_particles[j+1:]:
                    pairwise_distances.append(distance3D(other_central, cen_particle))


            count = len(central_particles)
            local_counts.append(count)
        # print(f"Rank {rank} finished boxes {local_offset}-{local_offset+len(local_boxes)} of frame {frame}")


        # Gather back to root
        frame_counts = comm.gather(local_counts, root=0)
        frame_distances = comm.gather(pairwise_distances, root=0)

        if rank == 0:
            flat = [c for rank_result in frame_counts for c in rank_result]
            particles_near_center.append(flat)
            all_distances = [d for rank_d in frame_distances for d in rank_d]

    # Rank 0 plots results
    if rank == 0:
        flat_counts = [c for frame in particles_near_center for c in frame]

        counts_per_n = list(range(0, max(flat_counts)))
        for n in flat_counts:
            counts_per_n[n-1] += 1
        for i in range(len(counts_per_n)):
            counts_per_n[i] /= len(flat_counts)

        # fig2, ax2 = plt.subplots(figsize=(8, 5), tight_layout=True)
        # ax2.hist(all_distances, bins=30, color='teal', edgecolor='black', alpha=0.75)
        # ax2.set_xlabel("Distance (Å)", fontsize=12)
        # ax2.set_ylabel("Frequency", fontsize=12)
        # ax2.set_title("Pairwise Distances Near Partition Centers", fontsize=14)
        # ax2.grid(True, linestyle='--', alpha=0.6)
        # plt.savefig("DistanceHistogram.png", format='png')

        fig, ax = plt.subplots(figsize=(8, 5), tight_layout=True)

        density_ratios = []
        inner_volume = (4/3)*math.pi*pow(radius_from_center, 3)
        for count in flat_counts:
            density_ratios.append((count/inner_volume)/bulk_density)
        n, bins, patches = ax.hist(
            density_ratios,
            bins=30,
            color='#4a90e2',
            edgecolor='black',
            alpha=0.85,
            rwidth=1
        )
        ax.set_xlabel(f"Relative Density within {radius_from_center} Å of partition center", fontsize=12)
        ax.set_ylabel("Number of partitions", fontsize=12)
        ax.set_title("Relative Density of Partition Centers", fontsize=14)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        for count, x in zip(n, bins[:-1]):
            if count > 0:
                ax.text(x + (bins[1] - bins[0]) / 2, count, str(int(count)), ha='center', va='bottom', fontsize=10)
        plt.savefig("WaterDensity.png", format='png')
        plt.show()

        with open("particles_near_center.csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Frame", *[f"Box{i}" for i in range(partitions)]])
            for frame_idx, frame in enumerate(particles_near_center):
                writer.writerow([frame_idx] + frame)