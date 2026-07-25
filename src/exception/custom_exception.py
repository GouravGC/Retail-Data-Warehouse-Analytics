"""
Custom exception handling module.

Provides a base custom exception class that captures
system-level error details for more informative debugging.
"""

import sys
import traceback


class CustomException(Exception):
    """
    Custom exception wrapper that captures the original error,
    file name, and line number where the exception occurred.
    """

    def __init__(self, error_message: str, error_detail: sys = sys):
        super().__init__(error_message)
        self.error_message = self._format_error_message(
            error_message, error_detail
        )

    @staticmethod
    def _format_error_message(
        error_message: str, error_detail: sys
    ) -> str:
        """
        Extract traceback info and format a detailed error string.

        Args:
            error_message: The original error message.
            error_detail: The sys module containing traceback info.

        Returns:
            A formatted string with file name, line number, and message.
        """
        _, _, exc_tb = error_detail.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        return (
            f"Error occurred in file: [{file_name}] "
            f"at line: [{line_number}] - {error_message}"
        )

    def __str__(self):
        return self.error_message

    def __repr__(self):
        return f"CustomException({self.error_message!r})"

