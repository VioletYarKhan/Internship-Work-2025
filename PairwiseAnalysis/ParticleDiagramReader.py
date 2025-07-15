import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import MDAnalysis as md
import math
from mpi4py import MPI
import csv
import argparse

# Splits an array into a 2D array containing num_splits equal-sized arrays if 
# num_splits%len(lst) == 0, otherwise it splits it into num_splits - 1 len(lst)//num_splits
# length arrays with a trailing len(lst)%num_splits length array
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

# Converts a linear index to 3D (x, y, z) coordinates based on the number of bins in each direction
def index_to_xyz(index, x_bins, y_bins):
    z = index // (x_bins * y_bins)
    remainder = index % (x_bins * y_bins) 
    y = remainder // x_bins
    x = remainder % x_bins
    return [x, y, z]

# Returns the center coordinates of a box given its index and binning parameters
def center_of_box(index, x_bins, y_bins, bins_per_axis, box_size):
    boxcoords = index_to_xyz(index, x_bins, y_bins)
    return [((pos + 0.5) * (box_size / bins_per_axis)) for pos in boxcoords]

# Returns the euclidian distance between 2 points in 3D space
def distance3D(coord1, coord2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(coord1, coord2)))

if __name__ == "__main__":
    # Initialize argparser
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--psf", help="The PSF file for the simulation", required=True)
    parser.add_argument("-d", "--dcd", help="The DCD file for the simulation", required=True)
    parser.add_argument("-s", "--psize", help="The requested partition size in Angstroms", type=float)
    parser.add_argument("-r", "--radius", help="The radius around the center of each partition to use in analysis", type=float, required = True)
    parser.add_argument("-b", "--bins-per-axis", help="The number of partitions per axis", type=int)

    args = parser.parse_args()

    if (args.psize is None and args.bins_per_axis is None) or (args.psize is not None and args.bins_per_axis is not None):
        parser.error("You must specify either --psize or --bins-per-axis, not neither or both.")

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Load simulation data
    PSF = args.psf
    DCD = args.dcd
    sim = md.Universe(PSF, DCD)

    # Determine box and partitioning parameters
    box_size = sim.trajectory[0].dimensions[0]
    if args.psize:
        partition_size_wanted = args.psize
        bins_per_axis = round(box_size/partition_size_wanted)
    if args.bins_per_axis:
        bins_per_axis = args.bins_per_axis

    x_bins = bins_per_axis
    y_bins = bins_per_axis
    z_bins = bins_per_axis
    partitions = x_bins * y_bins * z_bins
    partition_size = box_size / bins_per_axis
    radius_from_center = args.radius

    # Ensure partition size is large enough for the chosen radius
    assert partition_size >= 2 * radius_from_center, (f"Partition size is {partition_size} cubic angstroms, which is less than 2r ({2 * radius_from_center}).")

    if rank == 0:
        print(f'{bins_per_axis} bins per axis of {partition_size} cubic angstroms')

    nframes = sim.trajectory.n_frames
    oxygens = sim.select_atoms('name OH2')

    # Calculate bulk density of oxygens
    bulk_density = len(oxygens)/pow(box_size, 3)

    particles_near_center = []

    # Loop over all frames in the trajectory
    for frame in range(nframes):
        sim.trajectory[frame]

        # Shift coordinates for safety (center box at origin)
        for atom in oxygens:
            atom.position += box_size / 2

        # Assign each oxygen atom to a box
        boxes = [[] for _ in range(partitions)]
        for particle in oxygens.positions:
            xID = int((particle[0] / box_size) * x_bins)
            yID = int((particle[1] / box_size) * y_bins)
            zID = int((particle[2] / box_size) * z_bins)

            # Clamp indices to valid range
            xID = min(max(xID, 0), x_bins - 1)
            yID = min(max(yID, 0), y_bins - 1)
            zID = min(max(zID, 0), z_bins - 1)
            boxIndex = xID + yID * x_bins + zID * x_bins * y_bins
            boxes[boxIndex].append(particle)

        # Rank 0 splits the boxes for parallel processing
        if rank == 0:
            box_chunks = array_split(boxes, size)
        else:
            box_chunks = None

        # Distribute boxes among ranks
        local_boxes = comm.scatter(box_chunks, root=0)

        # Calculate local offset for global indexing
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
            # Separate out particles near the center
            for particle in box:
                if distance3D(particle, center) <= radius_from_center:
                    central_particles.append(particle)
                else:
                    noncentral_particles.append(particle)

            # Calculate pairwise distances between central and noncentral particles
            for j, cen_particle in enumerate(central_particles):
                for particle in noncentral_particles:
                    pairwise_distances.append(distance3D(particle, cen_particle))
                for other_central in central_particles[j+1:]:
                    pairwise_distances.append(distance3D(other_central, cen_particle))

            # Count number of central particles in this box
            count = len(central_particles)
            local_counts.append(count)

        # Gather results from all ranks
        frame_counts = comm.gather(local_counts, root=0)
        frame_distances = comm.gather(pairwise_distances, root=0)

        if rank == 0:
            flat = [c for rank_result in frame_counts for c in rank_result]
            particles_near_center.append(flat)
            all_distances = [d for rank_d in frame_distances for d in rank_d]

    # Rank 0 plots and saves results
    if rank == 0:
        flat_counts = [c for frame in particles_near_center for c in frame]

        # print(len(flat_counts))

        # Calculate histogram of counts per partition
        counts_per_n = list(range(0, max(flat_counts)))
        for n in flat_counts:
            counts_per_n[n-1] += 1
        for i in range(len(counts_per_n)):
            counts_per_n[i] /= len(flat_counts)
        
        # Calculate density ratios for each partition
        density_ratios = []
        inner_volume = (4/3)*math.pi*pow(radius_from_center, 3)
        for count in flat_counts:
            density_ratios.append((count/inner_volume)/bulk_density)
        
        print(f"Bulk Density: {bulk_density}")
        print(f"Frames: {nframes}")
        # print(f"Center Area: {partitions*inner_volume*nframes}")
        # print(f"Num of center particles: {sum(flat_counts)}")
        # print(f"Estimate of center particles: {partitions*inner_volume*nframes*bulk_density}")

        # Plot histogram of pairwise distances
        fig, ax = plt.subplots(figsize=(8, 5), tight_layout=True)
        ax.hist(all_distances, bins=30, color='teal', edgecolor='black', alpha=0.75)
        ax.set_xlabel("Distance (Å)", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title("Pairwise Distances Near Partition Centers", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.savefig("DistanceHistogram.png", format='png')

        # Plot histogram of density ratios
        density_ratios = []
        inner_volume = (4/3)*math.pi*pow(radius_from_center, 3) # Volume of sphere with radius radius_from_center
        for count in flat_counts:
            density_ratios.append((count/inner_volume)/bulk_density)
        
        plt.clf()
        plt.cla()

        fig, ax = plt.subplots(figsize=(8, 5), tight_layout=True)
        
        ax.hist(density_ratios, color='teal', edgecolor='black', alpha=0.75)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlabel(f"Relative Density within {radius_from_center} Å of partition center", fontsize=12)
        ax.set_ylabel("Number of partitions", fontsize=12)
        ax.set_title("Relative Density of Partition Centers", fontsize=14)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
   
        plt.savefig("WaterDensity.png", format='png')
        plt.show()

        plt.cla()
        plt.clf()

        # Plot histogram of number of oxygens near partition centers
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
        ax.set_yscale('log')
        plt.savefig("WaterHistogram.png", format='png')

        # Save per-frame, per-partition counts to CSV
        with open("particles_near_center.csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Frame", *[f"Box{i}" for i in range(partitions)]])
            for frame_idx, frame in enumerate(particles_near_center):
                writer.writerow([frame_idx] + frame)
