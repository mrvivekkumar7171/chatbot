"""
Calculator tool for performing basic arithmetic operations.
"""
from langchain_core.tools import tool

@tool
def calculator(
    first_num: float,
    second_num: float,
    operation: str,
) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div.

    Args:
        first_num (float): The first number.
        second_num (float): The second number.
        operation (str): The operation to perform ('add', 'sub', 'mul', 'div').

    Returns:
        dict: A dictionary containing input arguments and the result, or an error message.
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }

    except TypeError as e:
        return {"error": f"Invalid input types provided: {e}"}
