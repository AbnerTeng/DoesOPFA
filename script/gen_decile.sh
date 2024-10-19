ff3_cls=("SH" "SM" "SL" "BH" "BM" "BL" "ALL")
perfs=("ret" "sr")

for perf in "${perfs[@]}"; do
    for cls in "${ff3_cls[@]}"; do
        python -m src.decile_table --label_type "P_MC_${cls}(y=max)" --weighted_method "equal_weight" --get_full --perf "$perf"
        python -m src.decile_table --label_type "P_MC_${cls}(y=max)" --weighted_method "value_weight" --get_full --perf "$perf"
    done
done