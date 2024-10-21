#!/bin/bash

# Check if the user provided the input file path
if [ -z "$1" ]; then
    echo "Usage: $0 <input_log_file>"
    exit 1
fi

# Input log file from the argument
input_log="$1"

# Output file (you can modify this if needed)
output_log="filtered_log.txt"

# Use sed to filter the relevant lines
sed -nE '
  # Match and format lines with the required info [1][ 50/5172] lr: 1.0000e-05 loss: 1.0665
  s/.*\[(.*)\]\[(.*)\].*lr: ([^ ]+) .* loss: ([^ ]+).*/[\1][\2]  lr: \3 loss: \4/p

  # Print lines that contain "bbox_mAP_copypaste"
  s/.*bbox_mAP_copypaste: (.*)/bbox_mAP_copypaste: \1/p
  s/.*accuracy\/top1: (.*)/accuracy\/top1: \1/p
' "$input_log" >"$output_log"

echo "Filtered log saved to $output_log"
