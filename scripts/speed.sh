#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --job-name=dpfn
#SBATCH --constraint=cpunode

source "/var/scratch/${USER}/projects/dpfn/scripts/preamble.sh"

echo `pwd`
echo "PYTHON: `which python`"
echo 'Starting'

python3 /var/scratch/${USER}/projects/dpfn_util/tests/speed_against_dpfn_fn_full.py

wait

echo 'Finished :)'
