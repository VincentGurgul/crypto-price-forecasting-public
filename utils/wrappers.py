''' Utils package with wrapper / decorator functions. '''

import time
import logging

from functools import wraps
from humanfriendly import format_timespan

from .telegram import sendMessage


def log_execution(func):
    ''' Sets up logger and logs beginning and end of function execution to a
    log file in the current working directory. '''

    # Logger configuration
    logging.basicConfig(filename='python.log',
                        format='%(levelname)s | %(asctime)s | %(message)s',
                        datefmt='%d.%m.%Y %H:%M:%S',
                        encoding='utf-8',
                        level=logging.DEBUG)
    
    # Wrapper that logs function execution
    @wraps(func)
    def wrapper(*args, **kwargs):
        logging.debug(f"Executing {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            logging.error(f'Function {func.__name__} raised error during execution: {type(e).__name__} ({e})')
            raise e
            
        logging.debug(f"Finished executing {func.__name__}")
        return result
    
    return wrapper


def timeit(func):
    ''' Prints time that it took a function to run after successfull execution. '''
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        print(f"Function {func.__name__} took {format_timespan(duration)} to run.")
        return result

    return wrapper


def telegram_notify(func):
    ''' Sends telegram message confirming successfull completion or informing
    of erroneous termination of function execution. '''
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            output = func(*args, **kwargs)
            sendMessage(f'Function {func.__name__} successfully executed \U0001F389')
            return output
        except Exception as e:
            sendMessage(f'Function {func.__name__} returned error: {e} \U0001F614')
            raise e
                
    return wrapper


def retry(max_tries: int = 3, delay_seconds: int = 1):
    ''' Tries to run a function 'max_tries' times with a 'delay_seconds' delay
    between runs.
    
    Args:
        max_tries (int, optional): defaults to 3
        delay_seconds (int, optional): defaults to 1
    '''
    def decorator(func):
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            tries = 0
            while tries < max_tries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    tries += 1
                    print(f'(Retry wrapper) Function {func.__name__} raised error during execution no. {tries}: {type(e).__name__} ({e})')
                    if tries == max_tries:
                        raise e
                    time.sleep(delay_seconds)
                    
        return wrapper
    
    return decorator
