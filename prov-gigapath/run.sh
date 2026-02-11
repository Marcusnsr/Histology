#!/bin/bash
#SBATCH --job-name=gigapath
#SBATCH -p gpu --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/home/%u/gigapath_%j.out
#SBATCH --error=/home/%u/gigapath_%j.err

SCRIPT_DIR="/home/dlf903/Histology_v2/prov-gigapath"
MOUNT_SCRIPT="/home/dlf903/Histology_v2/scripts/mount_erda.sh"
UNMOUNT_SCRIPT="/home/dlf903/Histology_v2/scripts/unmount_erda.sh"

trap "bash $UNMOUNT_SCRIPT" EXIT

echo "Running mount script..."
bash "$MOUNT_SCRIPT"

if [ $? -ne 0 ]; then
    echo "Mount script failed! Exiting job."
    exit 1
fi

cd /home/dlf903/Histology_v2/prov-gigapath || { echo "cd failed"; exit 1; }

source /home/dlf903/miniconda3/etc/profile.d/conda.sh
conda activate gigapath

echo "Using Python: $(which python)"
python --version

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
which nvidia-smi && nvidia-smi || echo "nvidia-smi not found"
python - <<'PY'
import torch
print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.version.cuda:", torch.version.cuda)
print("CUDA_VISIBLE_DEVICES:", __import__("os").environ.get("CUDA_VISIBLE_DEVICES"))
PY

python /home/dlf903/Histology_v2/prov-gigapath/04_embed_dicom_gigapath.py
#python /home/dlf903/Histology_v2/prov-gigapath/05_verify_slides.py