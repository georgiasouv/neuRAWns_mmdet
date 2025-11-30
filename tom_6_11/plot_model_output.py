import os
import torch
from utils.load_raw_img import *

# =================================================================
# Imports

det_checkpoint = torch.load('tom_6_11/best_coco_bbox_mAP_epoch_19.pth')
rawmod_checkpoint = torch.load('tom_6_11/exp003_checkpoint.pth')
filepath = '/cifs/Shares/Raw_Bayer_Datasets/ROD/raw/train/day-00000.raw'



img = load_raw(filepath)






