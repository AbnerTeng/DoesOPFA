#!/bin/bash

figures_dir="dat/img"
training_dir="dat/train_t/"
testing_dir="dat/test_t/"

mkdir -p $training_dir
mkdir -p $testing_dir
total_files=$(find "$figures_dir" -type f | wc -l)
current_file=0


for fig in "$figures_dir"/*; do
    ((current_file++))
    ym=$(echo "$fig" | grep -o "[0-9]\{6\}")

    if [ "$ym" \< "200100" ]; then
        mv "$fig" "$training_dir"
    elif [ "$ym" \> "200012" ]; then
        mv "$fig" "$testing_dir"
    fi
    progress=$((current_file * 100 / total_files))
    echo -ne "Progress: ["
    fill=$((progress))
    for ((i = 0; i < 100; i++)); do
        if [ "$i" -lt "$fill" ]; then
            echo -ne "#"
        else
            echo -ne " "
        fi
    done
    echo -ne "] $progress%\r"
done

echo -e "\nDone!"