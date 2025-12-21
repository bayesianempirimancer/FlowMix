import matplotlib.pyplot as plt
import numpy as np

def visualize_validation(npz_path):
    data = np.load(npz_path)
    points = data['points']
    gt_features = data['gt_features']
    pred_features = data['pred_features']
    responsibilities = data['responsibilities'] # (K+1, N)
    object_ids = data['object_ids']
    
    # New sampled data
    if 'sampled_points' in data:
        sampled_points = data['sampled_points']
        sampled_features = data['sampled_features']
        has_samples = True
    else:
        has_samples = False
    
    # Clip features to 0-1 for RGB
    gt_features = np.clip(gt_features, 0, 1)
    pred_features = np.clip(pred_features, 0, 1)
    if has_samples:
        sampled_features = np.clip(sampled_features, 0, 1)
    
    # Predicted labels (Argmax responsibility)
    pred_labels = np.argmax(responsibilities, axis=0)
    
    # Create plot (2x3 layout now if samples exist)
    rows = 2
    cols = 3 if has_samples else 2
    fig = plt.figure(figsize=(5*cols, 10))
    
    # 1. GT Features
    ax1 = fig.add_subplot(rows, cols, 1, projection='3d')
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c=gt_features, s=5)
    ax1.set_title("GT Features")
    ax1.set_xlim(-2, 2); ax1.set_ylim(-2, 2); ax1.set_zlim(-2, 2)
    
    # 2. Pred Features (Reconstruction)
    ax2 = fig.add_subplot(rows, cols, 2, projection='3d')
    ax2.scatter(points[:, 0], points[:, 1], points[:, 2], c=pred_features, s=5)
    ax2.set_title("Predicted Features (Recon)")
    ax2.set_xlim(-2, 2); ax2.set_ylim(-2, 2); ax2.set_zlim(-2, 2)
    
    # 3. Generative Samples (New!)
    if has_samples:
        print(f"Sampled Points: Shape {sampled_points.shape}")
        print(f"  Min: {sampled_points.min(axis=0)}")
        print(f"  Max: {sampled_points.max(axis=0)}")
        print(f"  NaNs: {np.isnan(sampled_points).sum()}")
        
        ax3 = fig.add_subplot(rows, cols, 3, projection='3d')
        ax3.scatter(sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2], c=sampled_features, s=5)
        ax3.set_title("Generative Reconstruction (Sampled)")
        # Auto scale or fixed? Fixed helps comparison.
        # Check if points are within bounds
        if np.abs(sampled_points).max() > 3.0:
             print("WARNING: Sampled points outside [-2, 2] box. Using auto-scale.")
        else:
             ax3.set_xlim(-2, 2); ax3.set_ylim(-2, 2); ax3.set_zlim(-2, 2)
    
    # 4. GT Labels
    ax4 = fig.add_subplot(rows, cols, cols+1, projection='3d')
    ax4.scatter(points[:, 0], points[:, 1], points[:, 2], c=object_ids, cmap='tab10', s=5)
    ax4.set_title("GT Object IDs")
    ax4.set_xlim(-2, 2); ax4.set_ylim(-2, 2); ax4.set_zlim(-2, 2)
    
    # 5. Pred Labels
    ax5 = fig.add_subplot(rows, cols, cols+2, projection='3d')
    ax5.scatter(points[:, 0], points[:, 1], points[:, 2], c=pred_labels, cmap='tab10', s=5)
    ax5.set_title("Predicted Segmentation")
    ax5.set_xlim(-2, 2); ax5.set_ylim(-2, 2); ax5.set_zlim(-2, 2)
    
    plt.tight_layout()
    plt.savefig(npz_path.replace('.npz', '.png'))
    print(f"Saved plot to {npz_path.replace('.npz', '.png')}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        visualize_validation(sys.argv[1])
