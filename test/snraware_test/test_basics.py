import snraware
from types import ModuleType


def test_basics() -> None:
    assert type(snraware) == ModuleType
