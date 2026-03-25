from mmengine.hooks import Hook
from mmdet.registry import HOOKS


@HOOKS.register_module()
class ClassMappingValidationHook(Hook):
    """Hook to validate and print class mapping at training start."""
    
    def __init__(self, classes, priority='VERY_HIGH'):
        super().__init__()
        self.classes = classes
        self.priority = priority
        
    def before_train(self, runner):
        """Print class mapping validation before training starts."""
        
        print("\n" + "="*80)
        print("CLASS MAPPING VALIDATION - EXP004")
        print("="*80)
        
        print("\n[INFO] ROD → COCO Class Mapping:")
        print(f"  ROD annotations will be remapped to these COCO classes:")
        for idx, class_name in enumerate(self.classes):
            print(f"    ROD class {idx} → COCO '{class_name}'")
        
        print(f"\n[INFO] Total classes in config: {len(self.classes)}")
        
        # Get model num_classes
        model = runner.model
        if hasattr(model, 'roi_head') and hasattr(model.roi_head, 'bbox_head'):
            num_classes = model.roi_head.bbox_head.num_classes
            print(f"[INFO] Model bbox_head.num_classes: {num_classes}")
            
            if num_classes != len(self.classes):
                print(f"[WARNING] Expected {len(self.classes)} COCO classes, got {num_classes}!")
        
        print("\n" + "="*80 + "\n")