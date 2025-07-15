import matplotlib.pyplot as plt
import MDAnalysis as md
import math
from mpi4py import MPI
import csv
import argparse

# For an array of length n that should be split into k sections, it returns n % k sub-arrays of size n//k + 1 and the rest of size n//k.
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
def distance3D(coord1, coord2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(coord1, coord2)))

# Returns the center coordinates of a box given its index and binning parameters
def index_to_xyz(index, x_bins, y_bins):
    z = index // (x_bins * y_bins)
    remainder = index % (x_bins * y_bins)
    y = remainder // x_bins
    x = remainder % x_bins
    return [x, y, z]

# Returns the euclidian distance between 2 points in 3D space
def center_of_box(index, x_bins, y_bins, bins_per_axis, box_size):
    boxcoords = index_to_xyz(index, x_bins, y_bins)
    return [((pos + 0.5) * (box_size / bins_per_axis)) for pos in boxcoords]

if __name__ == "__main__":
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Initialize argparser
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--psf", required=True)
    parser.add_argument("-d", "--dcd", required=True)
    parser.add_argument("-s", "--psize", type=float)
    parser.add_argument("-r", "--radius", type=float, required=True)
    parser.add_argument("-b", "--bins-per-axis", type=int)
    args = parser.parse_args()

    if not args.psize and not args.bins_per_axis:
        raise Exception("Partition size or bins per axis must be defined")

    overlap_mode = args.psize is not None and args.bins_per_axis is not None

    # Load simulation data
    PSF, DCD = args.psf, args.dcd
    sim = md.Universe(PSF, DCD)
    box_size = sim.trajectory[0].dimensions[0]
    radius = args.radius

    # Determine box and partitioning parameters
    if overlap_mode:
        psize = args.psize
        bins_per_axis = args.bins_per_axis
        spacing = box_size / bins_per_axis
        # Ensure partition size is large enough for the chosen radius
        assert psize >= 2 * radius, "Partition size must be >= 2 * radius."
    else:
        if args.psize:
            psize = args.psize
            bins_per_axis = round(box_size / psize)
        else:
            bins_per_axis = args.bins_per_axis
        spacing = box_size / bins_per_axis
        psize = spacing

    if rank == 0:
        print(f"{bins_per_axis} bins per axis with spacing {spacing} and partition size {psize}")

    oxygens = sim.select_atoms("name OH2")
    # Calculate bulk density of oxygens
    bulk_density = len(oxygens) / (box_size ** 3)
    nframes = sim.trajectory.n_frames

    all_distances = []
    particles_near_center = []

    if overlap_mode:
        box_centers = [
            [(xi + 0.5) * spacing, (yi + 0.5) * spacing, (zi + 0.5) * spacing]
            for xi in range(bins_per_axis)
            for yi in range(bins_per_axis)
            for zi in range(bins_per_axis)
        ]
    else:
        box_centers = list(range(bins_per_axis ** 3))

    box_chunks = array_split(box_centers, size)
    local_centers = comm.scatter(box_chunks, root=0)

    # Loop over all frames in the trajectory
    for frame in range(nframes):
        sim.trajectory[frame]
        # Shift coordinates for safety (center box at origin)
        for atom in oxygens:
            atom.position += box_size / 2
        positions = oxygens.positions.copy()
        local_counts = []
        local_dists = []

        for center in local_centers:
            if overlap_mode:
                center_point = center
            else:
                center_point = center_of_box(center, bins_per_axis, bins_per_axis, bins_per_axis, box_size)
            # Find particles within partition cube
            nearby_particles = [pos for pos in positions if all(abs(pos[i] - center_point[i]) <= psize / 2 for i in range(3))]
            central_particles = [p for p in nearby_particles if distance3D(p, center_point) <= radius]
            noncentral = [p for p in nearby_particles if distance3D(p, center_point) > radius]

            for j, cp in enumerate(central_particles):
                for np in noncentral:
                    local_dists.append(distance3D(cp, np))
                for oc in central_particles[j+1:]:
                    local_dists.append(distance3D(cp, oc))

            local_counts.append(len(central_particles))

        frame_counts = comm.gather(local_counts, root=0)
        frame_dists = comm.gather(local_dists, root=0)

        # Rank 0 splits the boxes for parallel processing
        if rank == 0:
            flat = [c for sub in frame_counts for c in sub]
            particles_near_center.append(flat)
            all_distances.extend([d for sub in frame_dists for d in sub])
    
    if rank == 0:
        flat_counts = [c for frame in particles_near_center for c in frame]
        inner_volume = (4/3) * math.pi * radius**3
        density_ratios = [(count / inner_volume) / bulk_density for count in flat_counts]

        fig, ax = plt.subplots(figsize=(8, 5), tight_layout=True)
        ax.hist(all_distances, bins=30, color='teal', edgecolor='black', alpha=0.75)
        ax.set_xlabel("Distance (Å)")
        ax.set_ylabel("Frequency")
        ax.set_title("Pairwise Distances Near Partition Centers")
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.savefig("DistanceHistogram.png")
        plt.clf()

        fig, ax = plt.subplots(figsize=(8, 5), tight_layout=True)
        ax.hist(density_ratios, color='teal', edgecolor='black', alpha=0.75)
        ax.set_xlabel(f"Relative Density within {radius} Å")
        ax.set_ylabel("Number of centers")
        ax.set_title("Relative Density in Partitions")
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        plt.savefig("WaterDensity.png")
        plt.clf()

        fig, ax = plt.subplots(figsize=(8, 5), tight_layout=True)
        n, bins, _ = ax.hist(flat_counts, bins='auto', color='#4a90e2', edgecolor='black', alpha=0.85)
        ax.set_xlabel(f"Oxygens within {radius} Å")
        ax.set_ylabel("Number of centers")
        ax.set_title("Distribution of Oxygens in Spheres")
        ax.set_yscale("log")
        plt.savefig("WaterHistogram.png")

        with open("particles_near_center.csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Frame", *[f"Center{i}" for i in range(len(box_centers))]])
            for frame_idx, frame in enumerate(particles_near_center):
                writer.writerow([frame_idx] + frame)