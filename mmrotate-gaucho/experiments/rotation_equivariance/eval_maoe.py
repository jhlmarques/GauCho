import os
import pandas as pd
import mmcv
from mmcv.ops import box_iou_rotated
from mmrotate.core.bbox.transforms import obb2poly, poly2obb
import numpy as np
import torch
import ast
import re
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--gt', type=str, default='./experiments/rotation_equivariance/results_ann_multirot.csv')
    parser.add_argument('--iou', nargs='+', type=float, default=[0.5])
    return parser.parse_args()

# Calculate MAOE over an image's annotations and outputted predictions 
def calculateMAOE(preds, gts, angle_version, iou_thr=0.5):
    aoe_sum = 0.0
    n_matches = 0
    
    gts = torch.from_numpy(gts).float()
    preds = torch.from_numpy(preds).float()

    # Convert to le135
    gts = obb2poly(gts, angle_version) 
    gts = poly2obb(gts, 'le90')
    preds = obb2poly(preds, angle_version)
    preds = poly2obb(preds, 'le90')

    ious = box_iou_rotated(preds, gts).numpy()
    ious_max = ious.max(axis=1)
    ious_argmax = ious.argmax(axis=1)
    # For each prediction, check if the highest overlapping box has IoU above a threshold
    # If it does, calculate absolute orientation error between the prediction and aforementioned box
    maoe_data = []
    for i, pred in enumerate(preds):
        if ious_max[i] >= iou_thr:
            matched_gt = gts[ious_argmax[i]]
            n_matches += 1
    
            pred_angle = pred[4]
            gt_angle = matched_gt[4]

            diff1 = abs(pred_angle - gt_angle)
            diff2 = torch.pi - diff1
            aoe = torch.min(diff1, diff2) * 57.32

            if aoe > 20.0:
                print(f'High AOE ({aoe.item()}) (GT_ANGLE={gt_angle * 57.32})(AR={matched_gt[2]/matched_gt[3]})')

            maoe_data.append((pred.numpy(), matched_gt.numpy(), aoe.item()))
        else:
            maoe_data.append((pred.numpy(), None, 0.0))

    return maoe_data

def str_array_to_numpy(arr_str):
    arr_str = re.sub(' +', ' ', arr_str)
    arr_str = arr_str.replace(' ', ',')
    arr_str = arr_str.replace('[,', '[')
    arr_list = ast.literal_eval(arr_str)
    return np.vstack(arr_list)

if __name__ == '__main__':
    args = parse_args()
    config = args.config
    gts_csv = args.gt
    iou_thresholds = args.iou

    config_basename_noext = os.path.splitext(os.path.basename(args.config))[0]
    preds_csv = os.path.join('./experiments/rotation_equivariance/detections', config_basename_noext, 'results_pred_multirot.csv')
    if not os.path.exists(preds_csv):
        print(f'Cannot find {preds_csv}. Try running the prediction collection script')
        exit()

    df_preds = pd.read_csv(preds_csv)
    df_gts = pd.read_csv(gts_csv)

    # Join prediction and gt dataframes
    df_dets = pd.merge(df_gts, df_preds, how='inner', on=['dataset_rotation', 'gt_filename'])

    config = mmcv.Config.fromfile(config)
    angle_version = config.angle_version

    # For each dataset rotation, rotate annotations and calculate AOE
    aoe_dfs = []
    df_gp_rotation = df_dets.groupby('dataset_rotation')

    for rotation, df_rotation in tqdm(df_gp_rotation):
        df_rotation_gp_filename = df_rotation.groupby('gt_filename')
        rot_maoe_data = []
        for iou_threshold in iou_thresholds:
            aoe_sum = 0.0
            n_matches = 0
            n_preds = 0
            n_gts = 0
            for i, (filename, df_filename) in enumerate(df_rotation_gp_filename):

                # Read predictions and targets from csv
                gt_bboxes = str_array_to_numpy(df_filename['gt_bboxes'].values[0])
                n_gts += len(gt_bboxes)
                
                if df_filename['pred_bboxes'].values[0] == '[]' or df_filename['pred_bboxes'].isnull().any():
                    continue
                pred_bboxes = str_array_to_numpy(df_filename['pred_bboxes'].values[0])
                n_preds += len(pred_bboxes)
                
                # Calculate AOE
                im_maoe_data = calculateMAOE(pred_bboxes, gt_bboxes, angle_version, iou_threshold)
                rot_maoe_data.extend([(rotation, filename, iou_threshold) + data for data in im_maoe_data])
                
        df_rotation_maoe = pd.DataFrame(rot_maoe_data, columns=['rotation', 'filename', 'iou', 'pred', 'matched_gt', 'aoe'])
        aoe_dfs.append(df_rotation_maoe)


    if not os.path.exists('./experiments/rotation_equivariance/outputs'):
        os.mkdir('./experiments/rotation_equivariance/outputs')
    output_dir = os.path.join('./experiments/rotation_equivariance/outputs', config_basename_noext)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_path = os.path.join(output_dir, 'aoe.csv')

    aoe_dfs = pd.concat(aoe_dfs)
    aoe_dfs.to_csv(output_path)
    

