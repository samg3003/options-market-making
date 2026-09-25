import math

from src.pricing.black_scholes import blackScholesPrice
from src.pricing.types import EuropeanOption, OptionType
from src.pricing.greeks import blackScholesVega
from scipy.optimize import brentq

def validateMarketPrice(option: EuropeanOption, marketPrice: float, spot: float, rate: float, dividendYield: float) -> None:
    if marketPrice < 0:
        raise ValueError("marketPrice must be non-negative")

    timeToExpiry = option.timeToExpiry
    discountedSpot = spot * math.exp(-dividendYield * timeToExpiry)
    discountedStrike = option.strike * math.exp(-rate * timeToExpiry)

    if option.optionType == OptionType.CALL:
        lowerBound = max(discountedSpot - discountedStrike, 0.0)
        upperBound = discountedSpot
    else:
        lowerBound = max(discountedStrike - discountedSpot, 0.0)
        upperBound = discountedStrike

    # A market price outside these bounds cannot be produced by Black-Scholes.
    if marketPrice < lowerBound or marketPrice >= upperBound:
        raise ValueError("marketPrice violates no-arbitrage bounds")



def priceError(option: EuropeanOption, marketPrice: float, spot: float, rate: float, dividendYield: float, volatility: float) -> float:
    # The implied volatility is the volatility where this pricing error reaches zero.
    return blackScholesPrice(option, spot=spot, rate=rate, dividendYield=dividendYield, volatility=volatility) - marketPrice


def impliedVolatilityBisection(
    option: EuropeanOption,
    marketPrice: float,
    spot: float,
    rate: float,
    dividendYield: float,
    lowerVolatility: float = 1e-6,
    upperVolatility: float = 5.0,
    tolerance: float = 1e-8,
    maxIterations: int = 100,
) -> float:

    if spot <= 0:
        raise ValueError("spot must be positive")

    if lowerVolatility <= 0:
        raise ValueError("lowerVolatility must be positive")

    if upperVolatility <= lowerVolatility:
        raise ValueError("upperVolatility must exceed lowerVolatility")

    validateMarketPrice(option, marketPrice, spot, rate, dividendYield)

    # Bisection requires the root to be bracketed by opposite-signed errors.
    lowerError = priceError(option, marketPrice, spot, rate, dividendYield, lowerVolatility)
    upperError = priceError(option, marketPrice, spot, rate, dividendYield, upperVolatility)

    if lowerError * upperError > 0:
        raise ValueError("marketPrice is not bracketed by the volatility bounds")

    for _ in range(maxIterations):
        midpoint = (lowerVolatility + upperVolatility) / 2
        midpointError = priceError(option, marketPrice, spot, rate, dividendYield, midpoint)

        # Once the pricing error is sufficiently small, we have found the IV.
        if abs(midpointError) < tolerance:
            return midpoint

        if lowerError * midpointError <= 0:
            upperVolatility = midpoint
            upperError = midpointError
        else:
            lowerVolatility = midpoint
            lowerError = midpointError

    raise ValueError("implied volatility solver did not converge")

from src.pricing.greeks import blackScholesVega


def impliedVolatilityNewton(
    option: EuropeanOption,
    marketPrice: float,
    spot: float,
    rate: float,
    dividendYield: float,
    initialVolatility: float = 0.20,
    tolerance: float = 1e-8,
    maxIterations: int = 100,
) -> float:
    if spot <= 0:
        raise ValueError("spot must be positive")

    if initialVolatility <= 0:
        raise ValueError("initialVolatility must be positive")

    validateMarketPrice(option, marketPrice, spot, rate, dividendYield)

    volatility = initialVolatility

    for _ in range(maxIterations):
        error = priceError(option, marketPrice, spot, rate, dividendYield, volatility)

        # Newton uses Vega as the slope of price with respect to volatility.
        if abs(error) < tolerance:
            return volatility

        vega = blackScholesVega(option, spot, rate, dividendYield, volatility)

        if vega <= 1e-12:
            raise ValueError("Vega is too small for Newton-Raphson")

        volatility -= error / vega

        if volatility <= 0:
            raise ValueError("Newton-Raphson produced a non-positive volatility")

    raise ValueError("implied volatility solver did not converge")


def impliedVolatilityBrent(
    option: EuropeanOption,
    marketPrice: float,
    spot: float,
    rate: float,
    dividendYield: float,
    lowerVolatility: float = 1e-6,
    upperVolatility: float = 5.0,
) -> float:
    validateMarketPrice(option, marketPrice, spot, rate, dividendYield)

    # Brent combines bracketing robustness with faster interpolation steps.
    lowerError = priceError(option, marketPrice, spot, rate, dividendYield, lowerVolatility)
    upperError = priceError(option, marketPrice, spot, rate, dividendYield, upperVolatility)

    if lowerError * upperError > 0:
        raise ValueError("marketPrice is not bracketed by the volatility bounds")

    return brentq(
        lambda volatility: priceError(option, marketPrice, spot, rate, dividendYield, volatility),
        lowerVolatility,
        upperVolatility,
    )