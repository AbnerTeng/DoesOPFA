#!/bin/bash

echo "Cleaning up old versions..."

directory="dat"
max_index=0

for file in "$directory"/full_kjx_metric_v*.csv; do
    index=$(echo $file | grep -oE 'full_kjx_metric_v[0-9]+' | grep -oE '[0-9]+$')

    if [[ $index =~ ^[0-9]+$ ]]; then
        index=$((10#$index))

        if ((index > max_index)); then
            max_index=$index
        fi

    fi
done

i=$((max_index))
read -p "Recent version: v$i. Delete all versions older than v$(($i-3))? (y/n) " -n 1 -r

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo
    for file in "$directory"/full_kjx_metric_v*.csv; do
        index=$(echo $file | grep -oE 'full_kjx_metric_v[0-9]+' | grep -oE '[0-9]+$')

        if [[ $index =~ ^[0-9]+$ ]]; then
            index=$((10#$index))

            if ((index <= i - 3)); then
                echo "Deleting $file"
                rm "$file"
            fi
        fi
    done
else
    echo
    echo "Aborted."
    exit 1
fi