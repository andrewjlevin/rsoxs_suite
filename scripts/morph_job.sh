#!/bin/bash 

#SBATCH --account=ucb349_asc1
#SBATCH --partition=amilan
#SBATCH --nodes=4
#SBATCH --ntasks=208
#SBATCH --time=00:10:00
#SBATCH --job-name=fipy_morph_gen     
#SBATCH --output=fipy_morph_gen.%j.out 
#SBATCH --error=fipy_morph_gen.%j.out  
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anle1278@colorado.edu

# load necessary modules
module purge
module load aocc/3.1.0 openmpi
module load anaconda

export SLURM_EXPORT_ENV=ALL

# enable conda environment
conda activate nrss

## RUN YOUR PROGRAM ##
# python /pl/active/Toney-group/anle1278/rsoxs_suite/scripts/fipy_morph_gen.py
OMP_NUM_THREADS=1 mpirun -np $SLURM_NTASKS python /pl/active/Toney-group/anle1278/rsoxs_suite/scripts/fipy_morph_gen.py --petsc
# mpirun -np $SLURM_NTASKS python /pl/active/Toney-group/anle1278/parallel.py
