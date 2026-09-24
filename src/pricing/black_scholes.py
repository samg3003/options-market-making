import math
from scipy.stats import norm
from src.pricing.types import EuropeanOption, OptionType

# calculates d1 and d2 for use in black scholes pricing and the greeks
def calculateD1D2(
    option: EuropeanOption,
    spot: float,
    rate: float,
    dividendYield: float,
    volatility: float,
) -> tuple[float, float]:
    if spot <= 0:
        raise ValueError("spot must be positive")

    if volatility <= 0:
        raise ValueError("volatility must be positive")

    if option.timeToExpiry <= 0:
        raise ValueError("timeToExpiry must be positive for d1 and d2")

    d1 = (
        math.log(spot / option.strike)
        + (
            rate
            - dividendYield
            + 0.5 * volatility ** 2
        ) * option.timeToExpiry
    ) / (
        volatility * math.sqrt(option.timeToExpiry)
    )

    d2 = d1 - volatility * math.sqrt(option.timeToExpiry)

    return d1, d2


def blackScholesPrice(
    option: EuropeanOption,
    spot: float,
    rate: float,
    dividendYield: float,
    volatility: float,
) -> float:
    if spot <= 0:
        raise ValueError("spot must be positive")

    if volatility < 0:
        raise ValueError("volatility must be non-negative")

    if option.timeToExpiry == 0:
        if option.optionType == OptionType.CALL:
            return max(spot - option.strike, 0.0)

        return max(option.strike - spot, 0.0)


    if volatility == 0:
        discountedSpot = spot * math.exp(
            -dividendYield * option.timeToExpiry
        )

        discountedStrike = option.strike * math.exp(
            -rate * option.timeToExpiry
        )

        if option.optionType == OptionType.CALL:
            return max(discountedSpot - discountedStrike, 0.0)

        return max(discountedStrike - discountedSpot, 0.0)
    

    d1, d2 = calculateD1D2(
        option,
        spot,
        rate,
        dividendYield,
        volatility,
    )

    discountedSpot = spot * math.exp(
        -dividendYield * option.timeToExpiry
    )

    discountedStrike = option.strike * math.exp(
        -rate * option.timeToExpiry
    )

    if option.optionType == OptionType.CALL:
        return (
            discountedSpot * norm.cdf(d1)
            - discountedStrike * norm.cdf(d2)
        )

    return (
        discountedStrike * norm.cdf(-d2)
        - discountedSpot * norm.cdf(-d1)
    )