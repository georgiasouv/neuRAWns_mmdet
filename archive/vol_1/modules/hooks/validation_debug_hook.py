from mmengine.hooks import Hook
from mmdet.registry import HOOKS
from collections import Counter

@HOOKS.register_module()
class ValidationDebugHook(Hook):
    def __init__(self):
        self.all_classes = []
        self.all_scores = []
        
    def before_val_epoch(self, runner):
        print("\n" + "="*70)
        print("VALIDATION DEBUG - STARTING")
        print("="*70)
        self.all_classes = []
        self.all_scores = []
        
    def after_val_iter(self, runner, batch_idx, data_batch, outputs):
        if isinstance(outputs, list) and len(outputs) > 0:
            for result in outputs:
                if hasattr(result, 'pred_instances'):
                    pred_instances = result.pred_instances
                    self.all_classes.extend(pred_instances.labels.cpu().tolist())
                    self.all_scores.extend(pred_instances.scores.cpu().tolist())
                    
    def after_val_epoch(self, runner, metrics):
        print("\n" + "="*70)
        print("VALIDATION DEBUG - COMPLETE")
        print("="*70)
        
        if self.all_classes:
            class_counts = Counter(self.all_classes)
            print(f"\nTotal predictions across all validation images: {len(self.all_classes)}")
            print(f"Class distribution:")
            for class_id in range(5):
                count = class_counts.get(class_id, 0)
                pct = 100 * count / len(self.all_classes) if self.all_classes else 0
                print(f"  Class {class_id}: {count:6d} predictions ({pct:5.1f}%)")
            
            print(f"\nScore statistics:")
            print(f"  Min: {min(self.all_scores):.4f}")
            print(f"  Max: {max(self.all_scores):.4f}")
            print(f"  Mean: {sum(self.all_scores)/len(self.all_scores):.4f}")
        
        print(f"\nMetrics: {metrics}")
        print("="*70 + "\n")