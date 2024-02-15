echo "Training Binary classification"
# python -m src.cnn_workstation_v2 --mode train --mdl bin --cls SH
# echo "bin SH training complete"
# python -m src.cnn_workstation_v2 --mode train --mdl bin --cls SM 
# echo "bin SM training complete"
# python -m src.cnn_workstation_v2 --mode train --mdl bin --cls SL
# echo "bin SL training complete"
# python -m src.cnn_workstation_v2 --mode train --mdl bin --cls BH
# echo "bin BH training complete"
# python -m src.cnn_workstation_v2 --mode train --mdl bin --cls BM 
# echo "bin BM training complete"
# python -m src.cnn_workstation_v2 --mode train --mdl bin --cls BL
# echo "bin BL training complete"
# python -m src.cnn_workstation_v2 --mode train --mdl bin --cls ALL
# echo "bin ALL training complete"
# echo "========================="
# echo "Training Multiclasss with k=10"
# python -m src.cnn_workstation_v2 --mode train --mdl multi_10 --cls ALL
# echo "multi ALL training complete"

echo "
Train on different class pre-trained model?
"
read -p "Enter y or n: " choice

if [ $choice == "y" ]; then
    echo "Train by bin_SH_best.ckpt..."
    python -m src.cnn_wokstation_v2 --mode test --cls SM --use_spec_mdl True --spec_mdl SH
    python -m src.cnn_wokstation_v2 --mode test --cls SL --use_spec_mdl True --spec_mdl SH 
    python -m src.cnn_wokstation_v2 --mode test --cls BM --use_spec_mdl True --spec_mdl SH
    python -m src.cnn_wokstation_v2 --mode test --cls BL --use_spec_mdl True --spec_mdl SH
    echo "Train by bin_SM_best.ckpt..."
    python -m src.cnn_wokstation_v2 --mode test --cls SH --use_spec_mdl True --spec_mdl SM
    python -m src.cnn_wokstation_v2 --mode test --cls SL --use_spec_mdl True --spec_mdl SM
    python -m src.cnn_wokstation_v2 --mode test --cls BH --use_spec_mdl True --spec_mdl SM
    python -m src.cnn_wokstation_v2 --mode test --cls BM --use_spec_mdl True --spec_mdl SM
    python -m src.cnn_wokstation_v2 --mode test --cls BL --use_spec_mdl True --spec_mdl SM
    echo "Train by bin_SL_best.ckpt..."
    python -m src.cnn_wokstation_v2 --mode test --cls SH --use_spec_mdl True --spec_mdl SL
    python -m src.cnn_wokstation_v2 --mode test --cls SM --use_spec_mdl True --spec_mdl SL
    python -m src.cnn_wokstation_v2 --mode test --cls BH --use_spec_mdl True --spec_mdl SL
    python -m src.cnn_wokstation_v2 --mode test --cls BM --use_spec_mdl True --spec_mdl SL
    python -m src.cnn_wokstation_v2 --mode test --cls BL --use_spec_mdl True --spec_mdl SL
    echo "Train by bin_BH_best.ckpt..."
    python -m src.cnn_wokstation_v2 --mode test --cls SH --use_spec_mdl True --spec_mdl BH
    python -m src.cnn_wokstation_v2 --mode test --cls SM --use_spec_mdl True --spec_mdl BH
    python -m src.cnn_wokstation_v2 --mode test --cls SL --use_spec_mdl True --spec_mdl BH
    python -m src.cnn_wokstation_v2 --mode test --cls BM --use_spec_mdl True --spec_mdl BH
    python -m src.cnn_wokstation_v2 --mode test --cls BL --use_spec_mdl True --spec_mdl BH
    echo "Train by bin_BM_best.ckpt..."
    python -m src.cnn_wokstation_v2 --mode test --cls SH --use_spec_mdl True --spec_mdl BM
    python -m src.cnn_wokstation_v2 --mode test --cls SM --use_spec_mdl True --spec_mdl BM
    python -m src.cnn_wokstation_v2 --mode test --cls SL --use_spec_mdl True --spec_mdl BM
    python -m src.cnn_wokstation_v2 --mode test --cls BH --use_spec_mdl True --spec_mdl BM
    python -m src.cnn_wokstation_v2 --mode test --cls BL --use_spec_mdl True --spec_mdl BM
    echo "Train by bin_BL_best.ckpt..."
    python -m src.cnn_wokstation_v2 --mode test --cls SH --use_spec_mdl True --spec_mdl BL
    python -m src.cnn_wokstation_v2 --mode test --cls SM --use_spec_mdl True --spec_mdl BL
    python -m src.cnn_wokstation_v2 --mode test --cls SL --use_spec_mdl True --spec_mdl BL
    python -m src.cnn_wokstation_v2 --mode test --cls BH --use_spec_mdl True --spec_mdl BL
    python -m src.cnn_wokstation_v2 --mode test --cls BM --use_spec_mdl True --spec_mdl BL
fi

