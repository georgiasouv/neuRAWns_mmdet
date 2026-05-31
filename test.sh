#!/bin/bash
#SBATCH --job-name=env_sanity
#SBATCH --partition=test
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/networkhome/WMGDS/souval_g/neuRAWns_mmdet/cluster_scripts/logs/sanity_%j.out
#SBATCH --error=/networkhome/WMGDS/souval_g/neuRAWns_mmdet/cluster_scripts/logs/sanity_%j.err

source /networkhome/WMGDS/souval_g/anaconda3/etc/profile.d/conda.sh
conda activate mmdet12

echo "=== Installing mmcv ==="
export CUDA_HOME=/networkhome/WMGDS/souval_g/anaconda3/envs/mmdet12
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.9 9.0"
MMCV_WITH_OPS=1 pip install mmcv==2.2.0 --no-binary mmcv

echo "=== Environment check ==="
python -c "
import torch
print('torch:', torch.__version__)
print('gpu:', torch.cuda.get_device_name(0))
from mmcv.ops import nms
print('mmcv NMS ok')
"