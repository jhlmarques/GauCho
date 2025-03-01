#!/bin/bash
if [ $# -eq 0 ]; then
    echo "No config directory"
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
output_dirs=()

printf "%-80s %-20s %s\n" "Collected configs" "Latest checkpoint" "Output dir"
for config in $config_dir/*$keyword*.py; do
    config_basename=$(basename "$config")
    config_basename_noext=$(basename "$config" .py)
    work_dir="./work_dirs/$config_basename_noext"
    output_dir="./work_dirs/test_$config_basename_noext"
    if [ ! -d "$work_dir" ]; then
        #echo -e "\tSkipping $config_basename - no work dir"
        continue
    fi
    last_checkpoint=$work_dir/latest.pth
    if [ ! -f $last_checkpoint ]; then
        #echo -e "\tSkipping $config_basename - no last_checkpoint file"
        continue
    fi 

    last_checkpoint=$(readlink -f $last_checkpoint)
    collected_configs+=($config)
    collected_checkpoints+=($last_checkpoint)
    output_dirs+=($output_dir)
    printf "%-80s %-20s %s\n" $config_basename $(basename $last_checkpoint .pth) $output_dir
done

read -p "Collect dota outputs (y/N)? " choice
case "$choice" in 
  y|Y ) collect_dota=true;;
  * ) collect_dota=false;;
esac

for ((i=0; i < ${#collected_configs[@]}; i++)); do
    config=${collected_configs[i]}
    checkpoint=${collected_checkpoints[i]}
    output_dir=${output_dirs[i]}
    config_basename_noext=$(basename "$config" .py)
    if $collect_dota; then
        python tools/test.py $config $checkpoint --work-dir $output_dir --format-only --eval-options submission_dir=$output_dir/collected_output
    else
        python tools/test.py $config $checkpoint --work-dir $output_dir --eval mAP
    fi
done

