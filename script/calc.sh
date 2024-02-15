#!/bin/bash

read -p "Select calculate measure (type the name): " measure

if [ $measure = "sharpe" ]; then

    echo "Calculate Sharpe Ratio"

    python - <<EOF
import pandas as pd
from src.utils.data_utils import calculate_sharpe
data = pd.read_csv("dat/every_profit.csv")
calculate_sharpe(data, "P_KJX_ALL_equal_weight", 0.002)
EOF

    echo "Sharpe Ratio calculated"
    # open dat/sharpe_ratio.csv
        
elif [ $measure = "alpha" ]; then

    echo "Calculate Alpha"

    python - << EOF
import pandas as pd
from src.utils.data_utils import get_alpha
data = pd.read_csv("dat/every_profit.csv")
get_alpha(data, "dat/ff5_dat.csv")
EOF

    echo "Alpha calculated"
    # open dat/alpha.csv
fi
