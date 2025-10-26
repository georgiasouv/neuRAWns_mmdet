#!/bin/bash
#SBATCH --job-name=exp002
#SBATCH --partition=xlong
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=cluster_scripts/logs/exp002_%j.out
#SBATCH --error=cluster_scripts/logs/exp002_%j.err

source /networkhome/WMGDS/souval_g/anaconda3/etc/profile.d/conda.sh
conda activate mmdet12
echo "=== Attempting Kerberos authentication ==="
kinit -r 7d -c FILE:/tmp/krb5cc_$(id -u) souval_g < ~/.kerberos_pass
kinit_status=$?
echo "kinit exit status: $kinit_status"
klist

echo "=== Starting training ==="
cd ~/neuRAWns_mmdet
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
mim train mmdet configs/raw_experiments/exp002.py --launcher none
