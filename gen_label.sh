echo "Label Generator"


echo "select data type
1. train
2. test
"

read -p "select data type (type the name): " dat_type

echo "select label type
1. bin (binary)
2. multi_3 (multi-class, k=3)
3. multi_10 (multi-class, k=10)
4. ova_h (one-vs-all with highest label as 1)
5. ova_l (one-vs-all with lowest label as 1)
"

read -p "select label type (type the short name): " label_type

label_list=("bin" "multi_3" "multi_10" "ova_h" "ova_l")

if [ $dat_type = "train" ]; then
    if [[ " ${label_list[@]} " =~ $label_type ]]; then
        python -m src.label_generator_v3 --dat_type $dat_type --label_type $label_type
        echo "Train data label generated"
    else
        echo "wrong label type"
    fi

elif [ $dat_type = "test" ]; then
    if [[ " ${label_list[@]} " =~ $label_type ]]; then
        python -m src.label_generator_v3 --dat_type $dat_type --label_type $label_type
        echo "Test data label generated"
    else
        echo "wrong label type"
    fi

else
    echo "wrong data type"
fi
