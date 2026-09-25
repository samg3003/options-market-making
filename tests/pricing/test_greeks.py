import math

from src.pricing.black_scholes import blackScholesPrice
from src.pricing.greeks import (
    blackScholesDelta,
    blackScholesGamma,
    blackScholesVega,
    blackScholesTheta,
    blackScholesRho,
)
from src.pricing.types import EuropeanOption, OptionType

def testAtTheMoneyCallDelta():
    option = EuropeanOption(strike=100.0, timeToExpiry=1.0, optionType=OptionType.CALL)

    delta = blackScholesDelta(option, spot=100.0, rate=0.05, dividendYield=0.0, volatility=0.20)

    assert abs(delta - 0.6368) < 0.001


def testAtTheMoneyPutDelta():
    option = EuropeanOption(strike=100.0, timeToExpiry=1.0, optionType=OptionType.PUT)

    delta = blackScholesDelta(option, spot=100.0, rate=0.05, dividendYield=0.0, volatility=0.20)

    assert abs(delta - (-0.3632)) < 0.001

def testDeltaMatchesFiniteDifference():
    option = EuropeanOption(strike=100.0, timeToExpiry=1.0, optionType=OptionType.CALL)
    spot = 100.0
    h = 0.01

    numericalDelta = (
        blackScholesPrice(option, spot=spot + h, rate=0.05, dividendYield=0.0, volatility=0.20)
        - blackScholesPrice(option, spot=spot - h, rate=0.05, dividendYield=0.0, volatility=0.20)
    ) / (2 * h)

    analyticalDelta = blackScholesDelta(option, spot=spot, rate=0.05, dividendYield=0.0, volatility=0.20)

    assert abs(analyticalDelta - numericalDelta) < 1e-6

def testAtTheMoneyCallGamma():
    option = EuropeanOption(strike=100.0, timeToExpiry=1.0, optionType=OptionType.CALL)

    gamma = blackScholesGamma(option, spot=100.0, rate=0.05, dividendYield=0.0, volatility=0.20)

    assert abs(gamma - 0.0188) < 0.001


def testCallAndPutHaveSameGamma():
    call = EuropeanOption(strike=100.0, timeToExpiry=1.0, optionType=OptionType.CALL)
    put = EuropeanOption(strike=100.0, timeToExpiry=1.0, optionType=OptionType.PUT)

    callGamma = blackScholesGamma(call, spot=100.0, rate=0.05, dividendYield=0.0, volatility=0.20)
    putGamma = blackScholesGamma(put, spot=100.0, rate=0.05, dividendYield=0.0, volatility=0.20)

    assert abs(callGamma - putGamma) < 1e-12

def testGammaMatchesFiniteDifference():
    option = EuropeanOption(strike=100.0, timeToExpiry=1.0, optionType=OptionType.CALL)
    spot = 100.0
    h = 0.01

    priceUp = blackScholesPrice(option, spot=spot + h, rate=0.05, dividendYield=0.0, volatility=0.20)
    price = blackScholesPrice(option, spot=spot, rate=0.05, dividendYield=0.0, volatility=0.20)
    priceDown = blackScholesPrice(option, spot=spot - h, rate=0.05, dividendYield=0.0, volatility=0.20)

    numericalGamma = (priceUp - 2 * price + priceDown) / h ** 2
    analyticalGamma = blackScholesGamma(option, spot=spot, rate=0.05, dividendYield=0.0, volatility=0.20)

    assert abs(analyticalGamma - numericalGamma) < 1e-6


def testVegaMatchesFiniteDifference():
    option = EuropeanOption(strike=100.0, timeToExpiry=1.0, optionType=OptionType.CALL)
    volatility = 0.20
    h = 1e-5

    priceUp = blackScholesPrice(option, spot=100.0, rate=0.05, dividendYield=0.0, volatility=volatility + h)
    priceDown = blackScholesPrice(option, spot=100.0, rate=0.05, dividendYield=0.0, volatility=volatility - h)

    numericalVega = (priceUp - priceDown) / (2 * h)
    analyticalVega = blackScholesVega(option, spot=100.0, rate=0.05, dividendYield=0.0, volatility=volatility)

    assert abs(analyticalVega - numericalVega) < 1e-6


def testThetaMatchesFiniteDifference():
    option = EuropeanOption(strike=100.0, timeToExpiry=1.0, optionType=OptionType.CALL)
    h = 1e-4

    price = blackScholesPrice(option, spot=100.0, rate=0.05, dividendYield=0.0, volatility=0.20)

    shorterOption = EuropeanOption(strike=100.0, timeToExpiry=1.0 - h, optionType=OptionType.CALL)
    priceShorter = blackScholesPrice(shorterOption, spot=100.0, rate=0.05, dividendYield=0.0, volatility=0.20)

    numericalTheta = (priceShorter - price) / h
    analyticalTheta = blackScholesTheta(option, spot=100.0, rate=0.05, dividendYield=0.0, volatility=0.20)

    assert abs(analyticalTheta - numericalTheta) < 1e-3


def testRhoMatchesFiniteDifference():
    option = EuropeanOption(strike=100.0, timeToExpiry=1.0, optionType=OptionType.CALL)
    rate = 0.05
    h = 1e-5

    priceUp = blackScholesPrice(option, spot=100.0, rate=rate + h, dividendYield=0.0, volatility=0.20)
    priceDown = blackScholesPrice(option, spot=100.0, rate=rate - h, dividendYield=0.0, volatility=0.20)

    numericalRho = (priceUp - priceDown) / (2 * h)
    analyticalRho = blackScholesRho(option, spot=100.0, rate=rate, dividendYield=0.0, volatility=0.20)

    assert abs(analyticalRho - numericalRho) < 1e-5