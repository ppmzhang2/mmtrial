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
mkdir -p "$DEST/bump"
mkdir -p "$DEST/crack"
mkdir -p "$DEST/depression"
mkdir -p "$DEST/displacement"
mkdir -p "$DEST/pothole"
mkdir -p "$DEST/vegetation"
mkdir -p "$DEST/uneven"

# Loop through all files in the source directory
for file in "$SOURCE"/*; do
    # Check if the file name contains "_bump_"
    if [[ $(basename "$file") == *_bump_* ]]; then
        cp "$file" "$DEST/bump/"

    # Check if the file name contains "_crack_"
    elif [[ $(basename "$file") == *_crack_* ]]; then
        cp "$file" "$DEST/crack/"

    # Check if the file name contains "_depression_"
    elif [[ $(basename "$file") == *_depression_* ]]; then
        cp "$file" "$DEST/depression/"

    # Check if the file name contains "_displacement_"
    elif [[ $(basename "$file") == *_displacement_* ]]; then
        cp "$file" "$DEST/displacement/"

    # Check if the file name contains "_pothole_"
    elif [[ $(basename "$file") == *_pothole_* ]]; then
        cp "$file" "$DEST/pothole/"

    # Check if the file name contains "_vegetation_"
    elif [[ $(basename "$file") == *_vegetation_* ]]; then
        cp "$file" "$DEST/vegetation/"

    # Check if the file name contains "_uneven_"
    elif [[ $(basename "$file") == *_uneven_* ]]; then
        cp "$file" "$DEST/uneven/"

    fi
done

echo "Files copied successfully based on the new name pattern."
