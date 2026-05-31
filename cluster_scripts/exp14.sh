#!/bin/bash
#SBATCH --job-name=exp14
#SBATCH --partition=test
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=/networkhome/WMGDS/souval_g/neuRAWns_mmdet/cluster_scripts/logs/exp14_%j.out
#SBATCH --error=/networkhome/WMGDS/souval_g/neuRAWns_mmdet/cluster_scripts/logs/exp14_%j.err

# ── Environment ───────────────────────────────────────────────
source /networkhome/WMGDS/souval_g/anaconda3/etc/profile.d/conda.sh
conda activate mmdet12

# ── Kerberos authentication ───────────────────────────────────
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

# ── Trigger CIFS mount ────────────────────────────────────────
echo "=== Triggering CIFS mount ==="
ls /cifs/Shares/WMGData/ > /dev/null 2>&1

# ── Training ──────────────────────────────────────────────────
echo "=== Starting exp14 ==="
cd /networkhome/WMGDS/souval_g/neuRAWns_mmdet
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

mim train mmdet configs/experiments/exp14_convgamma_k1.py \
    --launcher none \
    --work-dir /networkhome/WMGDS/souval_g/neuRAWns_mmdet/work_dirs/exp14 \
    --cfg-options \
        train_dataloader.dataset.data_root=/networkhome/WMGDS/souval_g/data/ROD/yolo/ \
        val_dataloader.dataset.data_root=/networkhome/WMGDS/souval_g/data/ROD/yolo/ \
        test_dataloader.dataset.data_root=/networkhome/WMGDS/souval_g/data/ROD/yolo/ \
        val_evaluator.ann_file=/networkhome/WMGDS/souval_g/data/ROD/yolo/raw/json_raw_coco_mapped/val.json \
        test_evaluator.ann_file=/networkhome/WMGDS/souval_g/data/ROD/yolo/raw/json_raw_coco_mapped/test.json \
        load_from=/networkhome/WMGDS/souval_g/checkpoints_mmdet/rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth

# ── Cleanup ───────────────────────────────────────────────────
kill $KRENEW_PID 2>/dev/null
echo "=== exp14 finished ==="
