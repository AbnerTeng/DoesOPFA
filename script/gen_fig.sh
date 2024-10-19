#!/bin/bash

# Check if dat/ exists
if [ ! -d "dat" ]; then
    echo "dat/ does not exists. Generating folder..."
    mkdir dat

# Generate figures for the paper
python -m src.gen_fig --do_sample False 

