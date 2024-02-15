# !/bin/bash
# weight_method=("equal_weight" "value_weight")
# ff3_cls=("SH" "SM" "SL" "BH" "BM" "BL")

# for weight in "${weight_method[@]}"; do
#     for cls in "${ff3_cls[@]}"; do
#         for mdl in "${ff3_cls[@]}"; do
#             if [ "$cls" != "$mdl" ]; then
#                 echo "Forming portfolio by $cls and $mdl using $weight"
#                 python -m src.portfolio_v2 --label_type "P_KJX_${cls}_by${mdl}" --weighted_method "$weight" --get_full
#             fi
#         done
#     done
# done

# echo "Generate Benchmark performance"
# weight_method=("equal_weight" "value_weight")
# ff3_cls=("SH" "SM" "SL" "BH" "BM" "BL")

# for weight in "${weight_method[@]}"; do
#     for cls in "${ff3_cls[@]}"; do
#         echo "Forming benchmark portfolio by $cls using $weight method"
#         python -m src.portfolio_v2 --label_type "P_KJX_${cls}" --weighted_method "$weight" --get_full
#         python -m src.portfolio_v2 --label_type "P_KJX_${cls}" --weighted_method "$weight" --gen_benchmark
#     done
# done

weight_method=("equal_weight" "value_weight")
ff3_cls=("SH" "SM" "SL" "BH" "BM" "BL")

for weight in "${weight_method[@]}"; do
    for cls in "${ff3_cls[@]}"; do
        python -m src.portfolio_v2 --label_type "P_MC_H_${cls}" --weighted_method "$weight" --get_full
        python -m src.portfolio_v2 --label_type "P_MC_L_${cls}" --weighted_method "$weight" --get_full
    done
done    


