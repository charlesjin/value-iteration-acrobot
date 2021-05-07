import multiprocessing as mp
import numpy as np
import ctypes

def shared_double_array(shape, lock=False):
    """
    Form a shared memory numpy array of doubles.
    
    http://stackoverflow.com/questions/5549190/is-shared-readonly-data-copied-to-different-processes-for-python-multiprocessing 
    https://stackoverflow.com/questions/38770681/sharing-a-ctypes-numpy-array-without-lock-when-using-multiprocessing
    """

    size = 1
    for dim in shape:
        size *= dim
    
    shared_array_base = mp.Array(ctypes.c_double, int(size), lock=lock)
    shared_array = np.ctypeslib.as_array(shared_array_base)
    shared_array = shared_array.reshape(*shape)
    return shared_array

