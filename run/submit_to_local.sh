#!/bin/bash

# example to run all commands in the specified parameter file on a local machine
# submit from project directory and activate conda environment first

date
uname -n

file=./run/highres_parameters_partial.tsv
numlines=$(wc -l < $file)


# run every line of the file
while IFS= read -r line; do
    # status message, how far are we out of total number lines
    echo "----------------------------------------"
    echo "Running line             $((++i)) / $numlines"
    echo "$line"
    $line
done < $file

date
