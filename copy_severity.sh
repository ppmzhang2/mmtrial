#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <source_folder> <dest_folder> <threshold>"
    exit 1
fi

# Assign input arguments to variables
SOURCE=$1
DEST=$2
THRESHOLD=$3

# Create destination subdirectories if they do not exist
mkdir -p "$DEST/fair"
mkdir -p "$DEST/poor"
mkdir -p "$DEST/verypoor"

# Regular expression pattern to capture bounding box information (_x, _y, _w, _h)
bbox_pattern='_x([0-9]+)_y([0-9]+)_w([0-9]+)_h([0-9]+)'

# Loop through all files in the source directory
for file in "$SOURCE"/*; do
    filename=$(basename "$file")

    # Extract bounding box information (x, y, w, h) using regex
    if [[ $filename =~ $bbox_pattern ]]; then
        # x=${BASH_REMATCH[1]}
        y=${BASH_REMATCH[2]}
        # w=${BASH_REMATCH[3]}
        h=${BASH_REMATCH[4]}

        # Output the bounding box information along with the severity
        # echo "File: $filename: Bounding Box: x=$x, y=$y, w=$w, h=$h"

        # Check if y + h is less than the threshold value (bounding boxes on
        # the top have fewer resolution and should be skipped)
        if ((y + h < THRESHOLD)); then
            echo "Skipping file: $filename as y + h < $THRESHOLD"
            continue
        fi
    else
        echo "Bounding box info not found in file: $filename"
    fi

    # Check if the file name contains "_fair_"
    if [[ $filename == *_fair_* ]]; then
        cp "$file" "$DEST/fair/"

    # Check if the file name contains "_poor_"
    elif [[ $filename == *_poor_* ]]; then
        cp "$file" "$DEST/poor/"

    # Check if the file name contains "_verypoor_"
    elif [[ $filename == *_verypoor_* ]]; then
        cp "$file" "$DEST/verypoor/"

    else
        echo "Severity not found in file: $filename"
    fi
done

echo "Files copied successfully based on the name pattern."
