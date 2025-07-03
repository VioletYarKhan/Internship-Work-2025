#!/bin/bash
#SBATCH --job-name=FluidAnalysis
#SBATCH --output=FluidDynamics_%j.out
#SBATCH --error=FluidDynamics_%j.err
#SBATCH --ntasks=4
#SBATCH --time=02:00:00
#SBATCH --nodes=2
#SBATCH -A trn039


module load miniforge3/23.11.0

srun -n 4 python ParticleDiagramReader.py
