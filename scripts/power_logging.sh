#!/bin/bash

# Define the output file
output_file="gpu_full_log.csv"

rm -f $output_file

# Lauch two commands in parallel, one is training a model, the other is logging GPU information.
stdbuf -oL nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit --format=csv -l 1 | while IFS= read -r line; do
    echo "$(date '+%Y-%m-%d %H:%M:%S'), $line"
done >> $output_file &
pid1=$!
python train.py --config config/ResNet50onEuroSAT/train.yaml &
pid2=$!

# Wait for both commands to finish
wait -n $pid1 $pid2

# Kill the remaining process
kill $pid1 $pid2

# Modify the first cell of the header to 'time'
sed -i '1s/^[^,]*/time/' $output_file