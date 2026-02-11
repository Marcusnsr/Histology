#!/bin/bash
#SBATCH --job-name=clean_data
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/%u/clean_data_%j.out
#SBATCH --error=/home/%u/clean_data_%j.err

# 1) Conda env
source /home/dlf903/miniconda3/etc/profile.d/conda.sh
conda activate gigapath

echo "Using Python: $(which python)"
python --version

# 2) Mount erda
source /home/dlf903/Histology_v2/scripts/mount_erda.sh

# 3) Run
python /home/dlf903/Histology_v2/src/label/03_clean_data.py