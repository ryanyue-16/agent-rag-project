from tools.calculator import calculator


def test_calculator_returns_correct_result() -> None:
    assert calculator("3 * (5 + 7)") == "36"


def test_calculator_returns_error_message() -> None:
    result = calculator("3 *")

    assert result.startswith("计算错误:")
