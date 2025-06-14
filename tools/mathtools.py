from langchain_core.tools import tool
import cmath


@tool
def multiply(a: float, b: float) -> float:
    """
    Multiplies two numbers.
    Args:
        a (float): the first number
        b (float): the second number
    """
    return a * b


@tool
def add(a: float, b: float) -> float:
    """
    Adds two numbers.
    Args:
        a (float): the first number
        b (float): the second number
    """
    return a + b


@tool
def subtract(a: float, b: float) -> int:
    """
    Subtracts two numbers.
    Args:
        a (float): the first number
        b (float): the second number
    """
    return a - b


@tool
def divide(a: float, b: float) -> float:
    """
    Divides two numbers.
    Args:
        a (float): the first float number
        b (float): the second float number
    """
    if b == 0:
        raise ValueError("Cannot divided by zero.")
    return a / b


@tool
def modulus(a: int, b: int) -> int:
    """
    Get the modulus of two numbers.
    Args:
        a (int): the first number
        b (int): the second number
    """
    return a % b


@tool
def power(a: float, b: float) -> float:
    """
    Get the power of two numbers.
    Args:
        a (float): the first number
        b (float): the second number
    """
    return a**b


@tool
def square_root(a: float) -> float | complex:
    """
    Get the square root of a number.
    Args:
        a (float): the number to get the square root of
    """
    if a >= 0:
        return a**0.5
    return cmath.sqrt(a)
