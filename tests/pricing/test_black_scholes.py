from src.pricing.black_scholes import blackScholesPrice, calculateD1D2
from src.pricing.types import EuropeanOption, OptionType
import math

def testAtTheMoneyCall():
    option = EuropeanOption(
        strike=100.0,
        timeToExpiry=1.0,
        optionType=OptionType.CALL,
    )

    price = blackScholesPrice(
        option,
        spot=100.0,
        rate=0.05,
        dividendYield=0.0,
        volatility=0.20,
    )

    assert abs(price - 10.4506) < 0.001


def testAtTheMoneyPut():
    option = EuropeanOption(
        strike=100.0,
        timeToExpiry=1.0,
        optionType=OptionType.PUT,
    )

    price = blackScholesPrice(
        option,
        spot=100.0,
        rate=0.05,
        dividendYield=0.0,
        volatility=0.20,
    )

    assert abs(price - 5.5735) < 0.001


def testPutCallParity():
    call = EuropeanOption(strike=100.0, timeToExpiry=1.0, optionType=OptionType.CALL)
    put = EuropeanOption(strike=100.0, timeToExpiry=1.0, optionType=OptionType.PUT)

    spot = 100.0
    rate = 0.05
    dividendYield = 0.02
    volatility = 0.20

    callPrice = blackScholesPrice(call, spot, rate, dividendYield, volatility)
    putPrice = blackScholesPrice(put, spot, rate, dividendYield, volatility)

    expectedDifference = spot * math.exp(-dividendYield) - 100.0 * math.exp(-rate)
    actualDifference = callPrice - putPrice

    assert abs(actualDifference - expectedDifference) < 1e-10

def testCallAtExpiry():
    option = EuropeanOption(strike=100.0, timeToExpiry=0.0, optionType=OptionType.CALL)

    price = blackScholesPrice(option, spot=110.0, rate=0.05, dividendYield=0.0, volatility=0.20)

    assert price == 10.0


def testPutAtExpiry():
    option = EuropeanOption(strike=100.0, timeToExpiry=0.0, optionType=OptionType.PUT)

    price = blackScholesPrice(option, spot=90.0, rate=0.05, dividendYield=0.0, volatility=0.20)

    assert price == 10.0

def testCallAtZeroVolatility():
    option = EuropeanOption(strike=100.0, timeToExpiry=1.0, optionType=OptionType.CALL)

    price = blackScholesPrice(option, spot=100.0, rate=0.05, dividendYield=0.0, volatility=0.0)

    expectedPrice = max(100.0 - 100.0 * math.exp(-0.05), 0.0)

    assert abs(price - expectedPrice) < 1e-10


def testPutAtZeroVolatility():
    option = EuropeanOption(strike=100.0, timeToExpiry=1.0, optionType=OptionType.PUT)

    price = blackScholesPrice(option, spot=100.0, rate=0.05, dividendYield=0.0, volatility=0.0)

    expectedPrice = max(100.0 * math.exp(-0.05) - 100.0, 0.0)

    assert abs(price - expectedPrice) < 1e-10

import pytest

def testRejectsNonPositiveSpot():
    option = EuropeanOption(strike=100.0, timeToExpiry=1.0, optionType=OptionType.CALL)

    with pytest.raises(ValueError):
        blackScholesPrice(option, spot=0.0, rate=0.05, dividendYield=0.0, volatility=0.20)


def testRejectsNegativeVolatility():
    option = EuropeanOption(strike=100.0, timeToExpiry=1.0, optionType=OptionType.CALL)

    with pytest.raises(ValueError):
        blackScholesPrice(option, spot=100.0, rate=0.05, dividendYield=0.0, volatility=-0.20)

def testRejectsZeroVolatilityForD1D2():
    option = EuropeanOption(strike=100.0, timeToExpiry=1.0, optionType=OptionType.CALL)

    with pytest.raises(ValueError):
        calculateD1D2(option, spot=100.0, rate=0.05, dividendYield=0.0, volatility=0.0)