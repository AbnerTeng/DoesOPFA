import pytest
from src.portfolio_v2 import Portfolio

config = {
    "dat_path": None,
    "label_type": "P_KJX_SH",
}

@pytest.fixture
def portfolio_instance() -> Portfolio:
    return Portfolio(**config)


