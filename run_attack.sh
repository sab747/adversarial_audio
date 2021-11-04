#!/bin/bash

if [ $# -lt 5 ]
    then
    echo "Usage: $(basename $0) dataset_dir ckpts_dir output_root limit max_iters test_size."
    exit 1
fi
dataset_dir=$1
ckpts_dir=$2
output_root=$3
limit=$4
max_iters=$5
test_size=$6


frozen_graph="$ckpts_dir/fully_trained_SC.pb"
labels_file="$ckpts_dir/output_labels.txt"

if [ ! -d $ckpts_dir ]; then
    echo "Checkpoints dir does not exist."
    exit 1
fi

if [ ! -f $frozen_graph ]; then
    echo "Frozen graph does not exist."
    exit 1
fi

if [ ! -f $labels_file ]; then
    echo "Labels file does not exist."
    exit 1
fi
labels_list=`cat $labels_file`
sample_size=100


# remove output root if exists
if [ -d $output_root ] ; then
    echo "Output root directory already exists -- replacing! "
    rm -rf $output_root
    # exit 1
fi
mkdir $output_root

# Copy data files to ~/$output_root/data
for label in `ls $dataset_dir`
do
#TODO(malzantot) : see why this line ignores the labels with __
    if [ -d "$dataset_dir/$label" ] && [[ $labels_list == *"$label"* ]]; then
        mkdir -p "$output_root/data/$label"
        find "$dataset_dir/$label/" -name "*.wav" | sort -R \
        | head -n$test_size | gxargs -L1 gcp -t "$output_root/data/$label"
    fi
done


export PYTHONWARNINGS=ignore
mkdir -p "$output_root/result/"
for target_label in `ls $output_root/data/`
do
    for source_label in `ls $output_root/data`
    do
        if [ $source_label == $target_label ]; then
            continue
        fi
        echo "Running attack: $source_label --> $target_label"
        output_dir="$output_root/result/$target_label/$source_label"
        mkdir -p $output_dir
        python -Wignore audio_attack.py \
        --data-dir "$output_root/data/$source_label" \
        --output-dir $output_dir \
        --target-label $target_label \
        --labels-path $labels_file \
        --graph-path $frozen_graph \
        --limit $limit \
        --max-iters $max_iters
    done
done
