#!/bin/bash
# weight_method=("value_weight" "vol_weight")
# ff3_cls=("SH" "SM" "SL" "BH" "BM" "BL")

# for weight in "${weight_method[@]}"; do
#     for cls in "${ff3_cls[@]}"; do
#         for mdl in "${ff3_cls[@]}"; do
#             if [ "$cls" != "$mdl" ]; then
#                 echo "Forming portfolio by class $cls and model $mdl using weight method $weight"
#                 python -m src.portfolio_v2 --label_type "P_KJX_${cls}_by${mdl}" --weighted_method "$weight"
#             fi
#         done
#     done
# done

echo "Generate Benchmark performance"
weight_method=("equal_weight" "value_weight" "vol_weight")
ff3_cls=("SH" "SM" "SL" "BH" "BM" "BL")

for weight in "${weight_method[@]}"; do
    for cls in "${ff3_cls[@]}"; do
        echo "Forming benchmark portfolio by $cls using $weight method"
        python -m src.portfolio_v2 --label "P_KJX_${cls}" --weighted_method "$weight"
        python -m src.portfolio_v2 --label "P_KJX_${cls}" --weighted_method "$weight" --gen_benchmark
    done
done