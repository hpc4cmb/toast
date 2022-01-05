from time import time

# TODO display funciton name in message
def get_compile_time(f):
    """
    Returns a transformed function that runs twice in order deduce and display its compile time
    """
    def f_timed(*args):
        # compile time + runtime
        start = time()
        f(*args)
        mid = time()
        # just runtime as compile time was cached
        result = f(*args)
        end = time()
        # computes the various times
        compile_plus_run_time = mid - start
        run_time = end - mid
        compile_time = max(0., compile_plus_run_time - run_time)
        # displays times and returns result
        print(f"DEBUG: compile-time:{compile_time} run-time:{run_time}")
        return result
    return f_timed