"""Lazy GPU adapters. Run each model in its own upstream software environment."""
import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def normalized_lcf_config(path):
    cfg = json.loads(Path(path).read_text())
    for name in ('detector2d', 'detector3d', 'frustum_detector'):
        if name == 'frustum_detector' and not cfg.get('use_detection_recovery', True):
            continue
        for key in ('cfg_path', 'checkpoint_path'):
            raw = cfg[name][key].replace('${REPO_ROOT}', str(ROOT))
            raw = os.path.expanduser(os.path.expandvars(raw))
            if '${' in raw or '$' in raw:
                raise ValueError(f'Unresolved variable in {name}.{key}: {raw}')
            p = Path(raw)
            if not p.is_absolute():
                p = Path(path).resolve().parent / p
            if not p.is_file():
                raise FileNotFoundError(f'{name}.{key}: {p}')
            cfg[name][key] = str(p.resolve())
    # Nested upstream setting takes priority; keep the single source of truth.
    if 'score_thr' not in cfg['detector2d'] and 'score_thr_2d' in cfg:
        cfg['detector2d']['score_thr'] = cfg['score_thr_2d']
    return cfg


class MVXBackend:
    def __init__(self, config, checkpoint, device):
        if not checkpoint or not Path(checkpoint).is_file():
            raise FileNotFoundError('Provide a local trained MVX-Net checkpoint with --checkpoint')
        from copy import deepcopy
        from mmdet3d.apis import init_model
        from mmdet3d.structures import get_box_type
        from mmengine.dataset import Compose, pseudo_collate
        self.model = init_model(config, checkpoint, device=device)
        self.model.eval()
        self.resolved_config = self.model.cfg.pretty_text
        ds = self.model.cfg.test_dataloader.dataset
        self.pipeline = Compose(deepcopy(ds.pipeline))
        self.box_type, self.box_mode = get_box_type(ds.box_type_3d)
        self.classes = list(self.model.dataset_meta['classes'])
        self.collate = pseudo_collate

    def predict(self, sample, calibration):
        # Build the pipeline ONCE. Do not call the upstream helper with a full
        # info file per frame: that helper indexes data_list[0] for a single input.
        data = dict(lidar_points=dict(lidar_path=sample['points'], num_pts_feats=4),
                    img_path=sample['image'], sample_idx=sample['frame_id'],
                    box_type_3d=self.box_type, box_mode_3d=self.box_mode,
                    cam2img=calibration.camera_to_image.astype(np.float32),
                    lidar2cam=calibration.lidar_to_camera.astype(np.float32),
                    lidar2img=(calibration.camera_to_image @ calibration.lidar_to_camera).astype(np.float32))
        output = self.model.test_step(self.collate([self.pipeline(data)]))[0]
        return output.pred_instances_3d


class LCFBackend:
    def __init__(self, config, checkpoint, device):
        if checkpoint:
            raise ValueError('LCF3D uses component checkpoints from its JSON config')
        sys.path.insert(0, str(ROOT / 'src'))
        from single_view_inference import LateFusionSingleViewInferencer
        cfg = normalized_lcf_config(config)
        self.resolved_config = cfg
        self.model = LateFusionSingleViewInferencer(cfg, device=device)
        self.classes = cfg['classes']
        self.device = device

    def predict(self, sample, calibration):
        import torch
        t = torch.as_tensor(calibration.lidar_to_camera, device=self.device, dtype=torch.float32)
        p = torch.as_tensor(calibration.camera_to_image, device=self.device, dtype=torch.float32)
        return self.model.predict(sample['image'], sample['points'], t, p)


def unpack_predictions(prediction):
    from mmdet3d.structures import LiDARInstance3DBoxes
    boxes = prediction.bboxes_3d
    if not isinstance(boxes, LiDARInstance3DBoxes):
        raise ValueError('Expected LiDARInstance3DBoxes; convert coordinates explicitly')
    return (boxes.tensor.detach().float().cpu().numpy()[:, :7],
            prediction.scores_3d.detach().float().cpu().numpy(),
            prediction.labels_3d.detach().cpu().numpy())
