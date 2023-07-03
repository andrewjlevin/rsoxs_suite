#!/bin/bash 

#SBATCH --account=ucb349_asc1
#SBATCH --partition=aa100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0:30:00
#SBATCH --gres=gpu:3
#SBATCH --job-name=CYRSOXS      
#SBATCH --output=cyrsoxs.%j.out 
#SBATCH --error=cyrsoxs.%j.out  
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anle1278@colorado.edu

# load necessary modules
module purge
module load cmake
module load cuda
module load anaconda

# enable conda environment
conda activate nrss

## RUN YOUR PROGRAM ##
echo "RUNNING ON GPU"${CUDA_VISIBLE_DEVICES}
CyRSoXS BHJ.hdf5
