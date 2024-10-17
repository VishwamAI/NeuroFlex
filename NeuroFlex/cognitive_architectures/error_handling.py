import functools
import logging

logging.basicConfig(level=logging.DEBUG)

def enhanced_error_handling(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {str(e)}")
            # Implement graceful error recovery here
            # For now, we'll just re-raise the exception
            raise
    return wrapper

def create_enhanced_error_handling():
    return enhanced_error_handling
