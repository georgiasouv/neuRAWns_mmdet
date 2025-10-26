#!/bin/bash
#SBATCH --job-name=exp001_baseline
#SBATCH --partition=xlong
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=cluster_scripts/logs/exp001_%j.out
#SBATCH --error=cluster_scripts/logs/exp001_%j.err

source /networkhome/WMGDS/souval_g/anaconda3/etc/profile.d/conda.sh
conda activate mmdet12
echo "=== Attempting Kerberos authentication ==="
kinit -r 7d -c FILE:/tmp/krb5cc_$(id -u) souval_g < ~/.kerberos_pass
kinit_status=$?
echo "kinit exit status: $kinit_status"
klist
echo "=== Checking CIFS access ==="
ls -l /cifs/Shares/Raw_Bayer_Datasets/ROD/json_isp/train.json

echo "=== Starting training ==="
cd ~/neuRAWns_mmdet
mim train mmdet configs/baselines/exp001_faster_rcnn_r50_fpn.py --launcher none
