#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <source_folder> <dest_folder>"
    exit 1
fi

# Assign input arguments to variables
SOURCE=$1
DEST=$2

# Create destination subdirectories if they do not exist
mkdir -p "$DEST/fair"
mkdir -p "$DEST/poor"
mkdir -p "$DEST/verypoor"

# Loop through all files in the source directory
for file in "$SOURCE"/*; do
    # Check if the file name contains "_fair_"
    if [[ $(basename "$file") == *_fair_* ]]; then
        cp "$file" "$DEST/fair/"

    # Check if the file name contains "_poor_"
    elif [[ $(basename "$file") == *_poor_* ]]; then
        cp "$file" "$DEST/poor/"

    # Check if the file name contains "_verypoor_"
    elif [[ $(basename "$file") == *_verypoor_* ]]; then
        cp "$file" "$DEST/verypoor/"

    fi
done

echo "Files copied successfully based on the name pattern."
