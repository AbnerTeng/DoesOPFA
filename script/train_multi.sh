read -p "Task (Train / Test): " task
read -p "Train all multi-class models? (y/n): " choice

classes=("SH" "SM" "SL" "BH" "BM" "BL" "ALL")

if [ $task = "Train"]; then
    if [ $choice = "y" ]; then
        for cls in "${classes[@]}"; do
            python -m src.main --mode "train" --mdl "multi_10" --cls $cls
            echo "bin $cls training complete"
        done
    else
        read -p "Class to train: " cls
        python -m src.main --mode "train" --mdl "multi_10" --cls $cls
    fi

else
    read -p "Target class to test: " cls
    read -p "testing model: " spec_mdl
    python -m src.main --mode "test" --mdl "multi_10" --cls $cls --use_spec_mdl --spec_mdl $spec_mdl
fi