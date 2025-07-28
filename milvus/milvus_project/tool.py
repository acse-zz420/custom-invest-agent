import time
import functools

def async_timer(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        print(f"###TIME####{func.__name__} took {end_time - start_time:.2f} seconds to execute.")
        return result

    return wrapper


def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"###TIME####{func.__name__} took {end_time - start_time:.2f} seconds to execute.")
        return result

    return wrapper