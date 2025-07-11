#!/bin/bash
#SBATCH --job-name=FluidAnalysis
#SBATCH --output=FluidDynamics_%j.out
#SBATCH --ntasks=80
#SBATCH --time=02:00:00
#SBATCH --nodes=10
#SBATCH -A trn039


module load miniforge3/23.11.0

srun -n 80 python ParticleDiagramReader.py

