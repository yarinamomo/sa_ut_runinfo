#!/bin/bash

# Define the directory you want to loop through
directory="/tmp/files"

# Use a for loop to iterate over the files in the directory
for file in "$directory"/*; do
    if [ -f "$file" ]; then
        echo "Runing pythia over: $file"
        doop -a 1-call-site-sensitive+heap -i $file -id ut1 --platform python_2 --single-file-analysis --tensor-shape-analysis --full-tensor-precision
    fi
done

