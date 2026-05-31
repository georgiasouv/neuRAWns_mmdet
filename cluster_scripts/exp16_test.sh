#!/bin/bash
#SBATCH --job-name=exp16_test
#SBATCH --partition=test
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/networkhome/WMGDS/souval_g/neuRAWns_mmdet/cluster_scripts/logs_test/exp16_test_%j.out
#SBATCH --error=/networkhome/WMGDS/souval_g/neuRAWns_mmdet/cluster_scripts/logs_test/exp16_test_%j.err

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

# ── Test ──────────────────────────────────────────────────────
echo "=== Starting exp16 test ==="
cd /networkhome/WMGDS/souval_g/neuRAWns_mmdet
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# !! UPDATE THIS: replace epoch_X with the actual best epoch number
# Run: ls work_dirs/exp16/best_coco_bbox_mAP_*.pth
CHECKPOINT=/networkhome/WMGDS/souval_g/neuRAWns_mmdet/work_dirs/exp16/best_coco_bbox_mAP_epoch_7.pth

mim test mmdet /networkhome/WMGDS/souval_g/neuRAWns_mmdet/configs/experiments/exp16_test.py     --launcher none     --checkpoint $CHECKPOINT     --work-dir /networkhome/WMGDS/souval_g/neuRAWns_mmdet/work_dirs/exp16_test     --cfg-options         test_dataloader.dataset.data_root=/networkhome/WMGDS/souval_g/data/ROD/yolo/         test_evaluator.ann_file=/networkhome/WMGDS/souval_g/data/ROD/yolo/raw/json_raw_coco_mapped/test.json         load_from=/networkhome/WMGDS/souval_g/checkpoints_mmdet/rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth

# ── Cleanup ───────────────────────────────────────────────────
kill $KRENEW_PID 2>/dev/null
echo "=== exp16 test finished ==="