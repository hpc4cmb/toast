
import numpy as np
cimport numpy as np

np.import_array()

ctypedef np.float64_t f64_t
ctypedef np.int64_t i64_t
ctypedef np.uint64_t u64_t

cdef extern from "pytoast.h":
    void generate_grv(u64_t size, u64_t offset, u64_t counter1, u64_t counter2, u64_t key1, u64_t key2, f64_t* rand_array)
    void generate_neg11rv(u64_t size, u64_t offset, u64_t counter1, u64_t counter2, u64_t key1, u64_t key2, f64_t* rand_array)
    void generate_01rv(u64_t size, u64_t offset, u64_t counter1, u64_t counter2, u64_t key1, u64_t key2, f64_t* rand_array)
    void generate_uint64rv(u64_t size, u64_t offset, u64_t counter1, u64_t counter2, u64_t key1, u64_t key2, u64_t* rand_array)


def cbrng_normal(u64_t size, u64_t offset, u64_t counter1, u64_t counter2, u64_t key1, u64_t key2, np.ndarray[f64_t, ndim=1] rand_array):
    """Returns an array of gaussian random variables based on the threefry2x64 counter-based random number generator"""
    size = size - size%2
    # Perform array boundary check
    if np.shape(rand_array)[0] < (size+offset):
        print("Specified size & offset is out of array bounds")
        return
    generate_grv(size, offset, counter1, counter2, key1, key2, <f64_t*>rand_array.data)
    return

def cbrng_uniform_01_f64(u64_t size, u64_t offset, u64_t counter1, u64_t counter2, u64_t key1, u64_t key2, np.ndarray[f64_t, ndim=1] rand_array):
    """Returns an array of uniform random variables in (0,1) based on the threefry2x64 counter-based random number generator"""
    size = size - size%2
    # Perform array boundary check
    if np.shape(rand_array)[0] < (size+offset):
        print("Specified size & offset is out of array bounds")
        return
    generate_01rv(size, offset, counter1, counter2, key1, key2, <f64_t*>rand_array.data)
    return

def cbrng_uniform_m11_f64(u64_t size, u64_t offset, u64_t counter1, u64_t counter2, u64_t key1, u64_t key2, np.ndarray[f64_t, ndim=1] rand_array):
    """Returns an array of uniform random variables in (0,1) based on the threefry2x64 counter-based random number generator"""
    size = size - size%2
    # Perform array boundary check
    if np.shape(rand_array)[0] < (size+offset):
        print("Specified size & offset is out of array bounds")
        return
    generate_neg11rv(size, offset, counter1, counter2, key1, key2, <f64_t*>rand_array.data)
    return

def cbrng_uniform_uint64(u64_t size, u64_t offset, u64_t counter1, u64_t counter2, u64_t key1, u64_t key2, np.ndarray[u64_t, ndim=1] rand_array):
    """Returns an array of natural random variables in (unsigned 64-bit integer) based on the threefry2x64 counter-based random number generator"""
    size = size - size%2
    # Perform array boundary check
    if np.shape(rand_array)[0] < (size+offset):
        print("Specified size & offset is out of array bounds")
        return
    generate_uint64rv(size, offset, counter1, counter2, key1, key2, <u64_t*>rand_array.data)
    return
