#!/bin/bash 

#SBATCH --account=ucb349_asc1
#SBATCH --partition=amilan
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=0:10:00
#SBATCH --job-name=fipy_morph_gen     
#SBATCH --output=testing/fipy_morph_gen.%j.out 
#SBATCH --error=testing/fipy_morph_gen.%j.out  

# load necessary modules
module purge
module load anaconda

# enable conda environment
conda activate nrss

## RUN YOUR PROGRAM ##
python /pl/active/Toney-group/anle1278/rsoxs_suite/morph_gen/fipy_morph_gen.py
