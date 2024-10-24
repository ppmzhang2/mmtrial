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
mkdir -p "$DEST/bump"
mkdir -p "$DEST/crack"
mkdir -p "$DEST/depression"
mkdir -p "$DEST/displacement"
mkdir -p "$DEST/pothole"
mkdir -p "$DEST/vegetation"
mkdir -p "$DEST/uneven"

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

    # Check if the file name contains "_bump_"
    if [[ $filename == *_bump_* ]]; then
        cp "$file" "$DEST/bump/"

    # Check if the file name contains "_crack_"
    elif [[ $filename == *_crack_* ]]; then
        cp "$file" "$DEST/crack/"

    # Check if the file name contains "_depression_"
    elif [[ $filename == *_depression_* ]]; then
        cp "$file" "$DEST/depression/"

    # Check if the file name contains "_displacement_"
    elif [[ $filename == *_displacement_* ]]; then
        cp "$file" "$DEST/displacement/"

    # Check if the file name contains "_pothole_"
    elif [[ $filename == *_pothole_* ]]; then
        cp "$file" "$DEST/pothole/"

    # Check if the file name contains "_vegetation_"
    elif [[ $filename == *_vegetation_* ]]; then
        cp "$file" "$DEST/vegetation/"

    # Check if the file name contains "_uneven_"
    elif [[ $filename == *_uneven_* ]]; then
        cp "$file" "$DEST/uneven/"

    fi
done

echo "Files copied successfully based on the new name pattern."
