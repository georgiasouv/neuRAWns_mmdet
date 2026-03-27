#!/bin/bash
#SBATCH --job-name=exp014
#SBATCH --partition=xlong
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=/networkhome/WMGDS/souval_g/neuRAWns_mmdet/cluster_scripts/logs/exp014_%j.out
#SBATCH --error=/networkhome/WMGDS/souval_g/neuRAWns_mmdet/cluster_scripts/logs/exp014_%j.err

source /networkhome/WMGDS/souval_g/anaconda3/etc/profile.d/conda.sh
conda activate mmdet12

echo "=== Attempting Kerberos authentication ==="
kinit -r 7d -c FILE:/tmp/krb5cc_$(id -u) souval_g < ~/.kerberos_pass
kinit_status=$?
echo "kinit exit status: $kinit_status"
klist

echo "=== Starting Kerberos renewal loop ==="
(
  while true; do
    sleep 14400
    kinit -R -c FILE:/tmp/krb5cc_$(id -u) 2>/dev/null
    if [ $? -ne 0 ]; then
      kinit -r 7d -c FILE:/tmp/krb5cc_$(id -u) souval_g < ~/.kerberos_pass 2>/dev/null
    fi
  done
) &
KRENEW_PID=$!
echo "Kerberos renewal loop started (PID: $KRENEW_PID)"

echo "=== Triggering CIFS mount ==="
ls /cifs/Shares/WMGData/ > /dev/null 2>&1

echo "=== Starting exp014 ==="
cd /networkhome/WMGDS/souval_g/neuRAWns_mmdet
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

mim train mmdet cconfigs/learnable_process_singleDet/exp14_config.py --launcher none --work-dir work_dirs/exp014

kill $KRENEW_PID 2>/dev/null