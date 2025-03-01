import mmcv
from mmcv.runner import load_checkpoint

from mmrotate.datasets.pipelines.transforms import PolyRandomRotate
from mmrotate.datasets.builder import ROTATED_PIPELINES
from mmrotate.models import build_detector
from mmdet.datasets import build_dataset, build_dataloader
from mmdet.utils import build_dp

import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
import argparse
import os

DEVICE = 'cuda'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--checkpoint', type=str, required=False)
    parser.add_argument('--range', '-r', type=int, default=360)
    parser.add_argument('--anns', action='store_true')
    return parser.parse_args()

# Create image rotation transform class
@ROTATED_PIPELINES.register_module()
class ImageRotate(PolyRandomRotate):

    def __init__(self,
                 angle=0.0,
                 version='le90'):
        super().__init__()
        self.angle = angle
        self.version = version

    def __call__(self, results):
        results['rotate'] = True
        h, w, c = results['img_shape']
        img = results['img']
        results['rotate_angle'] = angle

        image_center = np.array((w / 2, h / 2))
        abs_cos, abs_sin = abs(np.cos(angle / 180 * np.pi)), abs(np.sin(angle / 180 * np.pi))

        bound_w, bound_h = w, h

        self.rm_coords = self.create_rotation_matrix(image_center, angle, bound_h, bound_w)
        self.rm_image = self.create_rotation_matrix(image_center, angle, bound_h, bound_w, offset=-0.5)

        img = self.apply_image(img, bound_h, bound_w)
        results['img'] = img
        results['img_shape'] = (bound_h, bound_w, c)

        return results

def insert_ImageRotate_transform(pipeline, angle, angle_version):
    dict_transform = dict(
        type='ImageRotate',
        angle=angle,
        version=angle_version
        )

    pipeline = pipeline.copy()
    pipeline.insert(1, dict_transform)
    return pipeline

def insert_PolyRotate_transform(pipeline, angle, angle_version):
    dict_transform1 = dict(type='LoadAnnotations', with_bbox=True)
    dict_transform2 = dict(
        type='PolyRandomRotate', # Effectively a determined rotation by given angle
        mode='value',
        angles_range=[angle],
        rotate_ratio=1.0,
        version=angle_version
        )

    pipeline = pipeline.copy()
    pipeline.insert(1, dict_transform1)
    pipeline.insert(2, dict_transform2)
    # Also remove resize and collect gt_bboxes
    pipeline[3]['scale_factor'] = 1.0
    pipeline[3]['img_scale'] = None
    pipeline[3]['transforms'][-1]['keys'] = ['img', 'gt_bboxes', 'gt_labels']
    return pipeline

if __name__ == '__main__':

    # This avoids pandas storing ellipses
    np.set_printoptions(threshold=1000)
    
    args = parse_args()
    max_rotation = args.range
    
    config = args.config
    checkpoint = args.checkpoint
    config_basename_noext = os.path.basename(config)[:-3]
    if checkpoint is None:
        checkpoint = os.path.join('work_dirs/', config_basename_noext, 'latest.pth')

    device = DEVICE
    
    config = mmcv.Config.fromfile(config)
    config.model.pretrained = None

    # Load pretrained model
    model = build_detector(config.model, test_cfg=config.get('test_cfg'))
    model = build_dp(model, device=device, device_ids=[0])
    checkpoint = load_checkpoint(model, checkpoint, map_location=device)
    model.CLASSES = checkpoint['meta']['CLASSES']
    model.cfg = config
    model.to(device)
    model.eval()

    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)
    test_loader_cfg = {
        **test_dataloader_default_args,
        **config.data.get('test_dataloader', {})
    }

    angle_version = config.angle_version


    collected_data = []

    # If true, collect annotations only
    if args.anns:
        config.data.test.pipeline = insert_PolyRotate_transform(config.data.test.pipeline, 0, angle_version)
        for angle in tqdm(range(0, max_rotation, 1)):     
            # Create test dataset instance
            config.data.test.pipeline[2]['angles_range'] = [angle]
            dataset = build_dataset(config.data.test)
            data_loader = build_dataloader(dataset, **test_loader_cfg)
            
            for data in data_loader:
                
                img_metas = data['img_metas'][0].data[0]
                filename = img_metas[0]['ori_filename']
                gt_bboxes = data['gt_bboxes'][0].data[0][0].numpy()

                data = (angle, filename, gt_bboxes)
                collected_data.append(data)
    
        df = pd.DataFrame(collected_data, columns = ['dataset_rotation', 'gt_filename', 'gt_bboxes'])
        output_dir = f'experiments/rotation_equivariance/detections/{config_basename_noext}'
        os.makedirs(output_dir, exist_ok=True)
        print(f'Outputs will be stored at {output_dir}')
        df.to_csv(os.path.join(output_dir, 'results_ann_multirot.csv'))

    else:
        config.data.test.pipeline = insert_ImageRotate_transform(config.data.test.pipeline, 0, angle_version)
        for angle in tqdm(range(0, max_rotation, 1)):     
            # Create test dataset instance
            dataset = build_dataset(config.data.test)
            config.data.test.pipeline[1]['angle'] = angle
            data_loader = build_dataloader(dataset, **test_loader_cfg)
            
            for data in data_loader:
                
                # Get model output
                with torch.no_grad():
                    results = model(return_loss=False, rescale=True, **data)

                img_metas = data['img_metas'][0].data[0]

                filename = img_metas[0]['ori_filename']
                pred_bboxes = results[0][0]

                data = (angle, filename, pred_bboxes)
                collected_data.append(data)
    
        df = pd.DataFrame(collected_data, columns = ['dataset_rotation', 'gt_filename', 'pred_bboxes'])
        output_dir = f'experiments/rotation_equivariance/detections/{config_basename_noext}'
        os.makedirs(output_dir, exist_ok=True)
        print(f'Outputs will be stored at {output_dir}')
        df.to_csv(os.path.join(output_dir, 'results_pred_multirot.csv'))
            

