from src.pricing.black_scholes import blackScholesPrice
from src.pricing.implied_volatility import (
    impliedVolatilityBisection,
    impliedVolatilityNewton,
    impliedVolatilityBrent,
)
from src.pricing.types import EuropeanOption, OptionType


def testCallRoundTrip():
    option = EuropeanOption(strike=100.0, timeToExpiry=1.0, optionType=OptionType.CALL)
    marketPrice = blackScholesPrice(option, spot=100.0, rate=0.05, dividendYield=0.0, volatility=0.20)

    for solver in (impliedVolatilityBisection, impliedVolatilityNewton, impliedVolatilityBrent):
        impliedVolatility = solver(option, marketPrice, spot=100.0, rate=0.05, dividendYield=0.0)
        assert abs(impliedVolatility - 0.20) < 1e-6


def testPutRoundTrip():
    option = EuropeanOption(strike=100.0, timeToExpiry=1.0, optionType=OptionType.PUT)
    marketPrice = blackScholesPrice(option, spot=100.0, rate=0.05, dividendYield=0.0, volatility=0.20)

    for solver in (impliedVolatilityBisection, impliedVolatilityNewton, impliedVolatilityBrent):
        impliedVolatility = solver(option, marketPrice, spot=100.0, rate=0.05, dividendYield=0.0)
        assert abs(impliedVolatility - 0.20) < 1e-6

def testRoundTripAtDifferentVolatility():
    option = EuropeanOption(strike=110.0, timeToExpiry=0.5, optionType=OptionType.CALL)
    volatility = 0.35

    marketPrice = blackScholesPrice(option, spot=100.0, rate=0.03, dividendYield=0.01, volatility=volatility)

    impliedVolatility = impliedVolatilityBrent(
        option,
        marketPrice,
        spot=100.0,
        rate=0.03,
        dividendYield=0.01,
    )

    assert abs(impliedVolatility - volatility) < 1e-6

import pytest

from src.pricing.implied_volatility import (
    impliedVolatilityBisection,
    impliedVolatilityNewton,
    impliedVolatilityBrent,
)


def testRejectsInvalidCallPrice():
    option = EuropeanOption(strike=100.0, timeToExpiry=1.0, optionType=OptionType.CALL)

    for solver in (impliedVolatilityBisection, impliedVolatilityNewton, impliedVolatilityBrent):
        with pytest.raises(ValueError):
            solver(option, 150.0, spot=100.0, rate=0.05, dividendYield=0.0)