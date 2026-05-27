import os
import pickle
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class SaveValPredictionsHook(Hook):
    def __init__(self, save_dir='val_predictions', save_every_n_epochs=10):
        self.save_dir = save_dir
        self.save_every_n_epochs = save_every_n_epochs
        self._predictions = []

    def before_val_epoch(self, runner):
        # reset buffer at the start of each val run
        self._predictions = []

    def after_val_iter(self, runner, batch_idx, data_batch, outputs):
        # only collect every N epochs
        if (runner.epoch) % self.save_every_n_epochs != 0:
            return

        for data_sample in outputs:
            pred_instances = data_sample.pred_instances
            gt_instances   = data_sample.gt_instances

            self._predictions.append({
                'img_path':    data_sample.img_path,
                'ori_shape':   data_sample.ori_shape,
                'img_shape':   data_sample.img_shape,
                'pred_bboxes': pred_instances.bboxes.cpu().numpy().tolist(),
                'pred_scores': pred_instances.scores.cpu().numpy().tolist(),
                'pred_labels': pred_instances.labels.cpu().numpy().tolist(),
                'gt_bboxes':   gt_instances.bboxes.cpu().numpy().tolist(),
                'gt_labels':   gt_instances.labels.cpu().numpy().tolist(),
            })

    def after_val_epoch(self, runner):
        if (runner.epoch) % self.save_every_n_epochs != 0:
            return
        if not self._predictions:
            return

        os.makedirs(self.save_dir, exist_ok=True)
        save_path = os.path.join(self.save_dir, f'epoch_{runner.epoch}_preds.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(self._predictions, f)

        runner.logger.info(
            f'[SaveValPredictionsHook] Saved {len(self._predictions)} '
            f'predictions to {save_path}'
        )
        self._predictions = []  # free memory