#!/bin/bash
echo "Select class: "
options=(
    "org_ff3_equal" "org_ff3_value" "org_ff3_volume"
    "bySH_equal" "bySM_equal" "bySL_equal"
    "bySH_value" "bySM_value" "bySL_value"
    "bySH_volume" "bySM_volume" "bySL_volume"
    "byBH_equal" "byBM_equal" "byBL_equal"
    "byBH_value" "byBM_value" "byBL_value"
    "byBH_volume" "byBM_volume" "byBL_volume"
    "SH_diff_mdl" "SM_diff_mdl" "SL_diff_mdl"
    "Quit"
)
PS3="Enter the class type: "
select opt in "${options[@]}"; do
    case $opt in
        "Quit")
            echo "Existing the script"
            break
            ;;
        *)
            echo "Drawing $opt"
            python -m src.viz --cls $opt
            break
            ;;
    esac
done