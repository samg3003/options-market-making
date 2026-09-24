import pytest

from src.pricing.types import EuropeanOption, OptionType


def testCreateCall():
    option = EuropeanOption(
        strike=100.0,
        timeToExpiry=1.0,
        optionType=OptionType.CALL,
    )

    assert option.strike == 100.0
    assert option.timeToExpiry == 1.0
    assert option.optionType == OptionType.CALL


def testCreatePut():
    option = EuropeanOption(
        strike=100.0,
        timeToExpiry=1.0,
        optionType=OptionType.PUT,
    )

    assert option.optionType == OptionType.PUT


def testStrikeMustBePositive():
    with pytest.raises(ValueError):
        EuropeanOption(
            strike=0.0,
            timeToExpiry=1.0,
            optionType=OptionType.CALL,
        )


def testTimeToExpiryCannotBeNegative():
    with pytest.raises(ValueError):
        EuropeanOption(
            strike=100.0,
            timeToExpiry=-1.0,
            optionType=OptionType.CALL,
        )


def testZeroTimeToExpiryIsAllowed():
    option = EuropeanOption(
        strike=100.0,
        timeToExpiry=0.0,
        optionType=OptionType.CALL,
    )

    assert option.timeToExpiry == 0.0