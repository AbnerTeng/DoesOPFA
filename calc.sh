#!/bin/bash

read -p "Select calculate measure (type the name): " measure

if [ $measure = "sharpe" ]; then

    echo "Calculate Sharpe Ratio"

    python - <<EOF
from src.utils import calculate_sharpe
calculate_sharpe("dat/every_data.csv")
EOF

    echo "Sharpe Ratio calculated"
    open dat/sharpe_ratio.csv
        
elif [ $measure = "alpha" ]; then

    echo "Calculate Alpha"

    python - << EOF
from src.utils import get_alpha
get_alpha("dat/every_data.csv", "dat/ff5_dat.csv")
EOF

    echo "Alpha calculated"
    open dat/alpha.csv
fi
