import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add project root to path
sys.path.insert(0, '/home/souval_g_WMGDS.WMG.WARWICK.AC.UK/Desktop/neuRAWns_mmdet')

from modules.raw_preprocessors import FixedISP

def read_raw_24b(file_path, img_shape=(1856, 2880)):
    """Read 24-bit raw file."""
    BIT8 = 2 ** 8
    BIT16 = 2 ** 16
    
    raw_data = np.fromfile(file_path, dtype=np.uint8)
    raw_data = raw_data.astype(np.float32)
    raw_data = raw_data[0::3] + raw_data[1::3] * BIT8 + raw_data[2::3] * BIT16
    raw_data = raw_data.reshape(img_shape).astype(np.float32)
    return raw_data

def denormalize_imagenet(tensor):
    """Reverse ImageNet normalization for visualization."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(tensor.device)
    return tensor * std + mean

def visualize_pipeline(raw_path):
    """Visualize FixedISP processing stages."""
    
    # 1. Load RAW image
    print(f"Loading RAW image: {raw_path}")
    raw_img = read_raw_24b(raw_path, img_shape=(1856, 2880))
    raw_tensor = torch.from_numpy(raw_img).unsqueeze(0).unsqueeze(0).cuda().float()
    print(f"RAW shape: {raw_tensor.shape}, range: [{raw_tensor.min():.2f}, {raw_tensor.max():.2f}]")
    
    # 2. Initialize FixedISP
    isp = FixedISP(norm_threshold=0.99, gamma=2.2).cuda()
    isp.eval()
    
    # 3. Process through ISP
    with torch.no_grad():
        # Get intermediate outputs for visualization
        x = raw_tensor
        
        # After packing
        x_packed = isp.packing(x)
        print(f"After packing: {x_packed.shape}, range: [{x_packed.min():.2f}, {x_packed.max():.2f}]")
        
        # After normalization
        x_norm = isp.adaptive_norm(x_packed)
        print(f"After norm: {x_norm.shape}, range: [{x_norm.min():.2f}, {x_norm.max():.2f}]")
        
        # After AWB
        x_awb = isp.awb(x_norm)
        print(f"After AWB: {x_awb.shape}, range: [{x_awb.min():.2f}, {x_awb.max():.2f}]")
        
        # After gamma
        x_gamma = isp.gamma_correction(x_awb)
        print(f"After gamma: {x_gamma.shape}, range: [{x_gamma.min():.2f}, {x_gamma.max():.2f}]")
        
        # Final output (with ImageNet norm)
        x_final = isp(raw_tensor)
        print(f"Final output: {x_final.shape}, range: [{x_final.min():.2f}, {x_final.max():.2f}]")
        
        # Denormalize for visualization
        x_vis = denormalize_imagenet(x_final)
        print(f"Denormalized: {x_vis.shape}, range: [{x_vis.min():.2f}, {x_vis.max():.2f}]")
    
    # 4. Convert to numpy for plotting
    def to_numpy(tensor):
        img = tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        return np.clip(img, 0, 1)
    
    # 5. Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Processing stages
    axes[0, 0].imshow(to_numpy(x_packed))
    axes[0, 0].set_title('1. After Packing (Demosaic)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(to_numpy(x_norm))
    axes[0, 1].set_title('2. After P99 Normalization')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(to_numpy(x_awb))
    axes[0, 2].set_title('3. After Auto White Balance')
    axes[0, 2].axis('off')
    
    # Row 2: Final stages
    axes[1, 0].imshow(to_numpy(x_gamma))
    axes[1, 0].set_title('4. After Gamma Correction')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(to_numpy(x_vis))
    axes[1, 1].set_title('5. Final (Denormalized for Viz)')
    axes[1, 1].axis('off')
    
    # Channel histograms
    rgb = to_numpy(x_vis)
    axes[1, 2].hist(rgb[:, :, 0].ravel(), bins=50, color='red', alpha=0.5, label='R')
    axes[1, 2].hist(rgb[:, :, 1].ravel(), bins=50, color='green', alpha=0.5, label='G')
    axes[1, 2].hist(rgb[:, :, 2].ravel(), bins=50, color='blue', alpha=0.5, label='B')
    axes[1, 2].set_title('RGB Histogram')
    axes[1, 2].legend()
    
    plt.tight_layout()
    
    # Save
    output_path = 'fixed_isp_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    plt.show()

if __name__ == '__main__':
    # Example RAW file path - change this to your actual file
    raw_path = '/cifs/Shares/Raw_Bayer_Datasets/ROD/raw/train/day-00000.raw'
    
    # Check if file exists
    if not os.path.exists(raw_path):
        print(f"File not found: {raw_path}")
        print("Please update the raw_path variable with a valid .raw file path")
        sys.exit(1)
    
    visualize_pipeline(raw_path)