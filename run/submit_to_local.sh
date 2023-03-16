#!/bin/bash

# example to run all commands in the specified parameter file on a local machine
# submit from project directory and activate conda environment first

date
uname -n

file=./run/parameters.tsv
numlines=$(wc -l < $file)

# run every line of the file
while IFS= read -r line; do
    # status message, how far are we out of total number lines
    echo "----------------------------------------"
    echo "Running line             $((++i)) / $numlines"
    echo "$line"
    # if the line is a comment, dont run
    if [[ $line == \#* ]]; then
        echo "Skipping comment line"
        continue
    fi
    $line
done < $file
