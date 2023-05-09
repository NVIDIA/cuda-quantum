import _pycudaq

def new_observe(*args, **kwargs):
    return _pycudaq.observe(args,kwargs)
