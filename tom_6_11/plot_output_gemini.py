import torch
from mmdet.apis import init_detector, inference_detector
from utils.load_raw_img import load_raw
from utils.ROD_isp import aesthetic_pipeline
import mmcv
from mmdet.apis import DetInferencer
# =================================================================
# 1. Define your model and image paths
# =================================================================

config_file = '/home/souval_g_WMGDS.WMG.WARWICK.AC.UK/Desktop/neuRAWns_mmdet/configs/raw_experiments/exp002.py' 
checkpoint_file = '/home/souval_g_WMGDS.WMG.WARWICK.AC.UK/Desktop/neuRAWns_mmdet/tom_6_11/best_coco_bbox_mAP_epoch_19.pth'
filepath = '/cifs/Shares/Raw_Bayer_Datasets/ROD/raw/train/day-00000.raw'

# =================================================================
# 2. Build the Model
# =================================================================
print("Initializing MMDetection 3.x Inferencer...")
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# This one line replaces init_detector.
# It builds the model from the config and loads the weights.
inferencer = DetInferencer(
    model=config_file,
    weights=checkpoint_file,
    device=device
)
print("Inferencer initialized.")

# =================================================================
# 3. Load the Image
# =================================================================
print(f"Loading image: {filepath}")
# Your code is correct here (assuming you fix the name/args)
img = load_raw(filepath) # Or img = load_raw_img(filepath, raw_dims=(1856, 2880))
# img = aesthetic_pipeline(img)
print(f"Image loaded, shape: {img.shape}")

# =================================================================
# 4. Run Inference & Get Visualisation
# =================================================================
print("Running inference and generating visualisation...")
# This one call replaces 'inference_detector' AND 'model.show_result'
# 'return_vis=True' tells it to draw the bounding boxes on the image.
result = inferencer(img, return_vis=True)

# The 'result' dictionary contains 'predictions' and 'visualization'
# We just want the visualised image.
output_image = result['visualization'][0]

# =================================================================
# 5. Plot the Output
# =================================================================
print("Saving visualization...")
mmcv.imwrite(output_image, 'plot_output.jpg')
print("Done. Check 'plot_output.jpg'.")