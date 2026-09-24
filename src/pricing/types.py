from dataclasses import dataclass
from enum import Enum


# an enum class of restricted set of possible option types
class OptionType(Enum):
    CALL = "call"
    PUT = "put"


@dataclass(frozen=True)
class EuropeanOption:
    strike: float
    timeToExpiry: float
    optionType: OptionType

    # the dataclass creastes the object first and then this lets us perform validation
    def __post_init__(self) -> None:
        if self.strike <= 0:
            raise ValueError("strike must be positive")

        if self.timeToExpiry < 0:
            raise ValueError("timeToExpiry must be non-negative")