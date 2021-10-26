#!/bin/bash
if [ $# -lt 2 ]
then
	echo "Usage: $(basename $0) output_dir ckpts_dir"
	exit 1
fi
output_dir=$1
result_dir="$output_dir/result"
labels_file="$2/conv_actions_labels.txt"
graph_file="$2/conv_actions_frozen.pb"

python3 evaluate_attack.py --output-dir=$result_dir --labels-path=$labels_file --graph-path=$graph_file -v
