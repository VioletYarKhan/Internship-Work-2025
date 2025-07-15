#!/bin/bash
#SBATCH --job-name=FluidAnalysis
#SBATCH --output=FluidDynamics_%j.out
#SBATCH --error=FDErr.err
#SBATCH --ntasks=80
#SBATCH --time=02:00:00
#SBATCH --nodes=10
#SBATCH -A trn039


module load miniforge3/23.11.0

srun -n 80 python ParticleDiagramReader.py -p w32768.psf -d eql.50.dcd -s 10 -r 3.5 

