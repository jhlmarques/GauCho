#!/bin/bash

echo "Comparing FPS performance for GauCho and Baseline OBB Head"

benchmark () {
    config=$1;
    checkpoint=$2;
    python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 tools/analysis_tools/benchmark.py \
    $config \
    $checkpoint \
    --launcher pytorch \
    --repeat-num $3;
}


arr_configs=(
    # RoI Transformer
    "configs_baseline/gaucho_two_stage_baseline_hrsc/roi_transformer_r50_fpn_probiou_6x_hrsc_rr_le90.py"
    "configs_gaucho/gaucho_two_stage_hrsc/roi_transformer_r50_fpn_gaucho_probiou_6x_hrsc_rr_le90.py"
    # Retinanet ATSS
    "configs_baseline/gaucho_one_stage_baseline_hrsc/rotated_atss_obb_r50_fpn_probiou_6x_hrsc_rr_le90.py"
    "configs_gaucho/gaucho_one_stage_hrsc/rotated_atss_obb_r50_fpn_gaucho_probiou_6x_hrsc_rr_le90.py"
    # R3Det ATSS
    "configs_baseline/gaucho_one_stage_baseline_hrsc/r3det_atss_r50_fpn_probiou_6x_hrsc_rr_le90.py"
    "configs_gaucho/gaucho_one_stage_hrsc/r3det_atss_r50_fpn_gaucho_probiou_6x_hrsc_rr_le90.py"
    # FCOS
    "configs_baseline/gaucho_anchorless_baseline_hrsc/gaussian_fcos_r50_fpn_gwd_6x_hsrc_le90.py"
    "configs_gaucho/gaucho_anchorless_hrsc/gaussian_fcos_r50_fpn_gaucho_gwd_6x_hsrc_le90.py"
)

arr_checkpoints=(
    # RoI Transformer
    "work_dirs_cvpr2025_submission/2_roi_transformer_r50_fpn_probiou_6x_hrsc_rr_le90/epoch_72.pth"
    "work_dirs_cvpr2025_submission/1_roi_transformer_r50_fpn_gaucho_probiou_6x_hrsc_rr_le90/epoch_72.pth"
    # Retinanet ATSS
    "work_dirs_cvpr2025_submission/rotated_atss_obb_r50_fpn_probiou_6x_hrsc_rr_le90/epoch_72.pth"
    "work_dirs_cvpr2025_submission/rotated_atss_obb_r50_fpn_gaucho_probiou_6x_hrsc_rr_le90/epoch_72.pth"
    # R3Det ATSS
    "work_dirs_cvpr2025_submission/r3det_atss_r50_fpn_probiou_6x_hrsc_rr_le90/epoch_72.pth"
    "work_dirs_cvpr2025_submission/r3det_atss_r50_fpn_gaucho_probiou_6x_hrsc_rr_le90/epoch_72.pth"
    # FCOS
    "work_dirs_cvpr2025_submission/gaussian_fcos_r50_fpn_gwd_6x_hsrc_le90/epoch_72.pth"
    "work_dirs_cvpr2025_submission/gaussian_fcos_r50_fpn_gaucho_gwd_6x_hsrc_le90/epoch_72.pth"
)

mkdir -p ./fps_comparison_results

N_REPEATS=3

output="Results: (${N_REPEATS} Repeat tests)\n"
for i in "${!arr_configs[@]}"; do
    config=${arr_configs[$i]}
    checkpoint=${arr_checkpoints[$i]}
    outfile="./fps_comparison_results/$(basename $config .py)"
    
    benchmark $config $checkpoint $N_REPEATS > $outfile
    
    mean_fps=$(cat $outfile | grep -Poh 'fps: [0-9]+(\.[0-9]+)' | awk '{sum+=$2} END {print sum/NR}')
    mean_time=$(cat $outfile | grep -Poh 'image: [0-9]+(\.[0-9]+)' | awk '{sum+=$2} END {print sum/NR}')
    echo "Mean FPS: ${mean_fps}" >> $outfile
    echo "Mean Time: ${mean_time}" >> $outfile
    output+="${config}:\n"
    output+="Mean FPS: ${mean_fps}\n"
    output+="Mean Time: ${mean_time}\n"
done

echo -e $output
