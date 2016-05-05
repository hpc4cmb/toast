
import numpy as np
cimport numpy as np

np.import_array()

ctypedef np.float64_t f64_t
ctypedef np.float32_t f32_t
ctypedef np.int64_t i64_t
ctypedef np.uint64_t u64_t
ctypedef np.int32_t i32_t
ctypedef np.uint32_t u32_t
ctypedef np.int16_t i16_t
ctypedef np.uint16_t u16_t
ctypedef np.int8_t i8_t
ctypedef np.uint8_t u8_t

cdef extern from "pytoast.h":
    # put declarations for rng C functions here...
    pass


# Low-level python functions to call the C functions here...

