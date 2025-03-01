from packaging.version import parse
import pandas as pd
import mmcv
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse
import numpy as np
from draw import draw_bboxes_from_series, draw_ellipses_from_series

EPS = 1e-2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_csv', type=str, help='Input .csv generated using the oriented detection collecting scripts')
    parser.add_argument('path_images', type=str, help='Path to the source image directory')
    parser.add_argument('output_path', type=str, help='Path to output images')
    parser.add_argument('-t', type=float, default=20.0, dest='min_oe', help='Minimum Orientation error over detections for an image to be visualized')
    parser.add_argument('-f', type=str, dest='filter_filename', help='If passed, only generates visualizations for the image with this filename')
    parser.add_argument('-e', action='store_true', dest='show_ellipses', help='If True, show Oriented Ellipse representation')
    parser.add_argument('-r', action='store_true', dest='raw_visualization', help='If true, only shows detections')
    parser.add_argument('-d', action='store_true', dest='drop_unmatched', help='If true, only shows detections')
    parser.add_argument('-a', type=int, dest='rotation_angle', help='Specify an image rotation to visualize')

    return parser.parse_args()

def get_high_oe_detections(df, high_oe, filter_filename, rotation):
    if filter_filename is None:
        print(f'MAOE (All Images): {df["aoe"].dropna().mean()}')
        df_high_oe = df.loc[df['aoe'] >= high_oe][['filename', 'rotation']]
    else:
        df_high_oe = df.loc[(df['aoe'] >= high_oe) & (df['filename'] == filter_filename)][['filename', 'rotation']]
        image_maoe = df.loc[df['filename'] == filter_filename]['aoe'].dropna().mean()
        print(f'MAOE (Single Image): {image_maoe}')
    
    if rotation is not None:
        df_high_oe = df_high_oe.loc[df_high_oe['rotation'] == rotation]

    
    df = df.merge(df_high_oe, on=['filename', 'rotation']).sort_values('aoe', ascending=False)
    return df

def visualize_high_oe_detections(df, image_path, output_path, show_ellipses, raw_visualization, drop_unmatched):

    for (filename, rotation), df_gp in tqdm(df.groupby(['filename', 'rotation'])):
        filename_full = os.path.join(image_path, filename)
        img = mmcv.imread(filename_full).astype(np.uint8)
        img = mmcv.bgr2rgb(img)
        img = mmcv.imrotate(img, angle=-rotation)
        width, height = img.shape[1], img.shape[0]
        img = np.ascontiguousarray(img)
        fig = plt.figure(f'Detections for {filename}', frameon=False)
        dpi = fig.get_dpi()
        fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax = plt.gca()
        ax.axis('off')
        
        if drop_unmatched:
            # Remove predictions that don't match any GT
            df_gp = df_gp.dropna()

        preds = df_gp['pred']
        matched_gts = df_gp['matched_gt'].dropna()

        highest_oe = df_gp['aoe'].max()
        
        if show_ellipses:
            ax = draw_ellipses_from_series(matched_gts, ax, 'green')
            ax = draw_ellipses_from_series(preds, ax, 'blue')
        else:
            ax = draw_bboxes_from_series(matched_gts, ax, 'green')
            ax = draw_bboxes_from_series(preds, ax, 'blue')
        
        if not raw_visualization:
            props = dict(boxstyle='square', facecolor='white', alpha=1.0)
            textstr = '\n'.join((
                f'Image Rotation = {rotation}º',
                f'Highest AOE = {highest_oe :.2f}º'
                ))
            ax.text(0.05, 0.1, textstr, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props)
        
        plt.imshow(img)



        canvas = fig.canvas
        stream, _ = canvas.print_to_buffer()
        buffer = np.frombuffer(stream, dtype='uint8')
        img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)
        img = rgb.astype('uint8')
        img = mmcv.rgb2bgr(img)

        if show_ellipses:
            out_file = os.path.join(output_path, f'ellipse_oe_{highest_oe : .2f}_dets_{filename[:-3] + "png"}')
        else:
            out_file = os.path.join(output_path, f'rot{rotation}_oe_{highest_oe : .2f}_dets_{filename[:-3] + "png"}')

        mmcv.imwrite(img, out_file)
        plt.close()

if __name__ == '__main__':
    args = parse_args()
    input_csv = args.input_csv
    img_path = args.path_images
    output_path = args.output_path
    min_oe = args.min_oe
    filter_filename = args.filter_filename
    show_ellipses = args.show_ellipses
    raw_visualization = args.raw_visualization
    drop_unmatched = args.drop_unmatched
    rotation = args.rotation_angle

    df = pd.read_csv(input_csv)
    df_high_oe = get_high_oe_detections(df, min_oe, filter_filename, rotation)
    img_plots = visualize_high_oe_detections(df_high_oe, img_path, output_path, show_ellipses, raw_visualization, drop_unmatched)