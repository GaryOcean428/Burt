import re
import traceback
from typing import Optional


class AgentZeroException(Exception):
    """Base exception class for AgentZero project."""

    def __init__(
        self, message: str, original_exception: Optional[Exception] = None
    ):
        super().__init__(message)
        self.original_exception = original_exception


class ToolExecutionError(AgentZeroException):
    """Exception raised when a tool execution fails."""

    pass


class ConfigurationError(AgentZeroException):
    """Exception raised when there's a configuration issue."""

    pass


class APIError(AgentZeroException):
    """Exception raised when an API call fails."""

    pass


def format_error(e: Exception, max_entries: int = 2) -> str:
    """
    Format an exception for logging or display.

    Args:
        e (Exception): The exception to format.
        max_entries (int): Maximum number of traceback entries to include.

    Returns:
        str: Formatted error message with truncated traceback.
    """
    if isinstance(e, AgentZeroException) and e.original_exception:
        e = e.original_exception

    traceback_text = traceback.format_exc()
    lines = traceback_text.split("\n")

    file_indices = [
        i for i, line in enumerate(lines) if line.strip().startswith("File ")
    ]

    if file_indices:
        start_index = max(0, len(file_indices) - max_entries)
        trimmed_lines = lines[file_indices[start_index] :]
    else:
        return traceback_text

    error_message = ""
    for line in reversed(trimmed_lines):
        if re.match(r"\w+Error:", line):
            error_message = line
            break

    result = "Traceback (most recent call last):\n" + "\n".join(trimmed_lines)
    if error_message:
        result += f"\n\n{error_message}"

    return result


def handle_exception(e: Exception, context: str = "") -> str:
    """
    Handle an exception by logging it and returning a user-friendly message.

    Args:
        e (Exception): The exception to handle.
        context (str): Additional context about where the error occurred.

    Returns:
        str: A user-friendly error message.
    """
    error_message = format_error(e)

    if context:
        error_message = f"Error in {context}:\n{error_message}"

    # Log the error (you may want to use a proper logging system here)
    print(f"ERROR: {error_message}")

    # Return a user-friendly message
    return "An error occurred. Please try again or contact support if the problem persists."
