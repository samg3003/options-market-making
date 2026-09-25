import math
from scipy.stats import norm

from src.pricing.black_scholes import calculateD1D2
from src.pricing.types import EuropeanOption, OptionType


def blackScholesDelta(option: EuropeanOption, spot: float, rate: float, dividendYield: float, volatility: float) -> float:
    if spot <= 0:
        raise ValueError("spot must be positive")

    if volatility <= 0:
        raise ValueError("volatility must be positive")

    if option.timeToExpiry <= 0:
        raise ValueError("timeToExpiry must be positive")

    d1, _ = calculateD1D2(option, spot, rate, dividendYield, volatility)

    if option.optionType == OptionType.CALL:
        return math.exp(-dividendYield * option.timeToExpiry) * norm.cdf(d1)

    return math.exp(-dividendYield * option.timeToExpiry) * (norm.cdf(d1) - 1.0)


def blackScholesGamma(option: EuropeanOption, spot: float, rate: float, dividendYield: float, volatility: float) -> float:
    if spot <= 0:
        raise ValueError("spot must be positive")

    if volatility <= 0:
        raise ValueError("volatility must be positive")

    if option.timeToExpiry <= 0:
        raise ValueError("timeToExpiry must be positive")

    # Gamma is the rate at which delta changes with spot.
    d1, _ = calculateD1D2(option, spot, rate, dividendYield, volatility)
    normalDensity = norm.pdf(d1)

    return math.exp(-dividendYield * option.timeToExpiry) * normalDensity / (
        spot * volatility * math.sqrt(option.timeToExpiry)
    )

def blackScholesVega(option: EuropeanOption, spot: float, rate: float, dividendYield: float, volatility: float) -> float:
    if spot <= 0:
        raise ValueError("spot must be positive")

    if volatility <= 0:
        raise ValueError("volatility must be positive")

    if option.timeToExpiry <= 0:
        raise ValueError("timeToExpiry must be positive")

    # Vega measures sensitivity of option value to volatility.
    d1, _ = calculateD1D2(option, spot, rate, dividendYield, volatility)

    return spot * math.exp(-dividendYield * option.timeToExpiry) * norm.pdf(d1) * math.sqrt(option.timeToExpiry)


def blackScholesTheta(option: EuropeanOption, spot: float, rate: float, dividendYield: float, volatility: float) -> float:
    if spot <= 0:
        raise ValueError("spot must be positive")

    if volatility <= 0:
        raise ValueError("volatility must be positive")

    if option.timeToExpiry <= 0:
        raise ValueError("timeToExpiry must be positive")

    d1, d2 = calculateD1D2(option, spot, rate, dividendYield, volatility)
    timeToExpiry = option.timeToExpiry
    discountedSpot = spot * math.exp(-dividendYield * timeToExpiry)
    discountedStrike = option.strike * math.exp(-rate * timeToExpiry)
    commonTerm = -discountedSpot * norm.pdf(d1) * volatility / (2 * math.sqrt(timeToExpiry))

    if option.optionType == OptionType.CALL:
        return commonTerm - dividendYield * discountedSpot * norm.cdf(d1) - rate * discountedStrike * norm.cdf(d2)

    return commonTerm + dividendYield * discountedSpot * norm.cdf(-d1) + rate * discountedStrike * norm.cdf(-d2)


def blackScholesRho(option: EuropeanOption, spot: float, rate: float, dividendYield: float, volatility: float) -> float:
    if spot <= 0:
        raise ValueError("spot must be positive")

    if volatility <= 0:
        raise ValueError("volatility must be positive")

    if option.timeToExpiry <= 0:
        raise ValueError("timeToExpiry must be positive")

    _, d2 = calculateD1D2(option, spot, rate, dividendYield, volatility)
    rhoScale = option.strike * option.timeToExpiry * math.exp(-rate * option.timeToExpiry)

    if option.optionType == OptionType.CALL:
        return rhoScale * norm.cdf(d2)

    return -rhoScale * norm.cdf(-d2)

