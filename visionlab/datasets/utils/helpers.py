import inspect

def has_keyword_arg(cls, arg_name):
    """
    Check if the constructor of a class takes a specific keyword argument.

    Parameters:
    cls (type): The class whose constructor to inspect.
    arg_name (str): The name of the keyword argument to look for.

    Returns:
    bool: True if the constructor has the specified keyword argument, False otherwise.
    """
    # Get the signature of the class's __init__ method
    try:
        signature = inspect.signature(cls.__init__)
    except ValueError:
        # Handle the case where __init__ isn't defined explicitly
        return False

    # Check if any parameter in the constructor matches the keyword argument name
    for param in signature.parameters.values():
        if param.name == arg_name and (
            param.default != inspect.Parameter.empty or param.kind == inspect.Parameter.KEYWORD_ONLY
        ):
            return True

    return False