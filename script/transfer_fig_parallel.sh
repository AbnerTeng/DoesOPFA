#!/bin/bash

figures_dir="fig/img"
training_dir="img/train_t"
testing_dir="img/test_t"

mkdir -p "$training_dir"
mkdir -p "$testing_dir"

total_files=$(find "$figures_dir" -type f | wc -l)
current_file=0

move_images() {
    fig="$1"
    ym=$(echo "$fig" | grep -o "[0-9]\{6\}")

    if [ "$ym" -lt "200100" ]; then
        mv "$fig" "$training_dir"
    elif [ "$ym" -gt "200012" ]; then
        mv "$fig" "$testing_dir"
    fi

    ((current_file++))
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
}

export -f move_images

find "$figures_dir" -type f | parallel move_images

echo -e "\nDone!"
