#!/bin/bash
if [ $# -eq 0 ]; then
    exit
fi

config_dir=$1
if [ ! -d "$config_dir" ]; then
    echo "$config_dir" is not a directory
    exit
fi

keyword=$2
if [ ! -z $keyword ]; then
    echo "Filtering configs containing keyword \"$keyword\""
fi

collected_configs=()
collected_checkpoints=()
echo "Collecting configs..."

for config in $config_dir/*$keyword*.py; do
    collected_configs+=($config)
    config_basename=$(basename "$config")
    config_basename_noext=$(basename "$config" .py)
    work_dir=work_dirs/$config_basename_noext
    if [ -d $work_dir ]; then
        last_checkpoint=$work_dir/latest.pth
        if [ -f $last_checkpoint ]; then
            last_checkpoint=$(readlink -f $last_checkpoint)
            collected_checkpoints+=("--resume-from $last_checkpoint")
            collect_msg=" - Work dir already exists (Checkpoint: $(basename $last_checkpoint))"
        else
            collected_checkpoints+=("")
            collect_msg=" - Work dir already exists (No checkpoint)"
        fi
    else
        collect_msg=" - No work dir"
    fi
 
    echo -e "\t$config_basename $collect_msg"
done

read -p "Press any key to continue" 

for ((i=0; i < ${#collected_configs[@]}; i++)); do
    config=${collected_configs[i]}
    checkpoint=${collected_checkpoints[i]}
    python tools/train.py $config $checkpoint --cfg-options checkpoint_config.interval=12 evaluation.interval=12
done

