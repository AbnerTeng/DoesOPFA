"""
pytest script
"""
import numpy as np
import pandas as pd
import pytest
from src.utils import data_utils


shift_config = {
    "data": pd.DataFrame(
        {
            "yyyymm": [202001, 202002, 202003, 202001, 202002, 202003],
            "PERMNO": [1, 1, 1, 2, 2, 2],
            "RET": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    ),
    "cols": "RET"
}

sharpe_config = {
    "data": pd.DataFrame(
        {
            "class1": [
                0.1, 0.1, 0.2, 0.2, 0.1, 0.2,
                0.3, 0.2, 0.1, 0.4, 0.2, 0.3
            ],
            "class2": [
                0.2, 0.2, -0.3, 0.3, -0.1, -0.2,
                0.3, 0.2, 0.4, -0.2, -0.1, 0.2
            ],
        }
    ),
    "rf": 0.00
}


def test_shift() -> None:
    assert data_utils.shift(**shift_config) == [0.2, 0.3, 0, 0.5, 0.6, 0]


def test_sharpe() -> None:
    assert np.round(
        data_utils.calculate_sharpe(**sharpe_config)["Annual Sharpe Ratio"].iloc[0], 2
    ) == 7.27
    assert np.round(
        data_utils.calculate_sharpe(**sharpe_config)["Annual Sharpe Ratio"].iloc[1], 2
    ) == 1.09


if __name__ == "__main__":
    pytest.main('-v', 'test_data_utils.py')
    
     